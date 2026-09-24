#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace {

double median(VectorXd values) {
    std::sort(values.data(), values.data() + values.size());
    const Eigen::Index n = values.size();

    if (n % 2 == 1) {
        return values(n / 2);
    }

    return 0.5 * (values(n / 2 - 1) + values(n / 2));
}

double sign_or_zero(double x) {
    if (x > 0.0) {
        return 1.0;
    }
    if (x < 0.0) {
        return -1.0;
    }
    return 0.0;
}

double time_weight(const std::optional<double>& half_life, py::ssize_t n_periods, py::ssize_t t) {
    if (!half_life.has_value()) {
        return 1.0;
    }
    const double age = static_cast<double>(n_periods - 1 - t);
    return std::exp2(-age / *half_life);
}

double mgs_weight(double x, double y, double gamma, double n) {
    const double numerator = std::sqrt((1.0 + std::abs(x)) * (1.0 + std::abs(y)));
    const double mismatch = std::pow(std::abs(std::abs(x) - std::abs(y)), n);
    return std::pow(numerator / (1.0 + mismatch), gamma);
}

VectorXd population_sds(const MatrixXd& mat) {
    const VectorXd mean_vec = mat.colwise().mean();
    VectorXd sd_vec(mat.cols());

    for (Eigen::Index col = 0; col < mat.cols(); ++col) {
        double sum_sq = 0.0;
        for (Eigen::Index row = 0; row < mat.rows(); ++row) {
            const double diff = mat(row, col) - mean_vec(col);
            sum_sq += diff * diff;
        }
        sd_vec(col) = std::sqrt(sum_sq / static_cast<double>(mat.rows()));
    }

    return sd_vec;
}

MatrixXd corr_to_cov(const MatrixXd& corr_mat, const VectorXd& sd_vec) {
    MatrixXd cov_mat = sd_vec.asDiagonal() * corr_mat * sd_vec.asDiagonal();
    cov_mat.diagonal() = sd_vec.array().square().matrix();
    return cov_mat;
}

MatrixXd center_matrix(const MatrixXd& mat, const std::string& center_method) {
    MatrixXd centered = mat;

    if (center_method == "mean") {
        centered.rowwise() -= mat.colwise().mean();
        return centered;
    }

    if (center_method == "median") {
        VectorXd median_vec(mat.cols());
        for (Eigen::Index col = 0; col < mat.cols(); ++col) {
            median_vec(col) = median(mat.col(col));
        }
        centered.rowwise() -= median_vec.transpose();
    }

    return centered;
}

MatrixXd array_to_matrix_2d(const py::array_t<double, py::array::c_style | py::array::forcecast>& array) {
    const auto view = array.unchecked<2>();
    MatrixXd mat(view.shape(0), view.shape(1));

    for (py::ssize_t row = 0; row < view.shape(0); ++row) {
        for (py::ssize_t col = 0; col < view.shape(1); ++col) {
            mat(row, col) = view(row, col);
        }
    }

    return mat;
}

std::vector<MatrixXd> parse_signal_input(const py::object& signal_input) {
    using Array = py::array_t<double, py::array::c_style | py::array::forcecast>;

    if (py::isinstance<py::array>(signal_input)) {
        const Array signal_array = signal_input.cast<Array>();
        const auto info = signal_array.request();

        if (info.ndim == 2) {
            return {array_to_matrix_2d(signal_array)};
        }

        if (info.ndim == 3) {
            const auto view = signal_array.unchecked<3>();
            std::vector<MatrixXd> signal_mats;
            signal_mats.reserve(view.shape(0));

            for (py::ssize_t m = 0; m < view.shape(0); ++m) {
                MatrixXd mat(view.shape(1), view.shape(2));
                for (py::ssize_t row = 0; row < view.shape(1); ++row) {
                    for (py::ssize_t col = 0; col < view.shape(2); ++col) {
                        mat(row, col) = view(m, row, col);
                    }
                }
                signal_mats.push_back(mat);
            }

            return signal_mats;
        }

        throw std::runtime_error(
            "signal_input must be a 2D matrix or a 3D array shaped (n_signals, n_periods, n_assets)."
        );
    }

    if (py::isinstance<py::sequence>(signal_input) && !py::isinstance<py::str>(signal_input)) {
        const py::sequence sequence = signal_input.cast<py::sequence>();
        if (sequence.size() == 0) {
            throw std::runtime_error("signal_input must contain at least one signal matrix.");
        }

        std::vector<MatrixXd> signal_mats;
        signal_mats.reserve(sequence.size());
        for (const py::handle item : sequence) {
            signal_mats.push_back(array_to_matrix_2d(item.cast<Array>()));
        }
        return signal_mats;
    }

    throw std::runtime_error("signal_input must be a 2D array, a 3D array, or a sequence of 2D arrays.");
}

VectorXd parse_volatility_stds(const py::object& volatility_source, Eigen::Index n_assets) {
    using Array = py::array_t<double, py::array::c_style | py::array::forcecast>;

    if (py::isinstance<py::array>(volatility_source)) {
        const Array vol_array = volatility_source.cast<Array>();
        const auto info = vol_array.request();

        if (info.ndim == 1) {
            const auto view = vol_array.unchecked<1>();
            if (view.shape(0) != n_assets) {
                throw std::runtime_error("volatility_source vector length must match the number of asset columns.");
            }

            VectorXd sd_vec(n_assets);
            for (py::ssize_t i = 0; i < view.shape(0); ++i) {
                sd_vec(i) = view(i);
            }
            return sd_vec;
        }

        if (info.ndim == 2) {
            const MatrixXd vol_mat = array_to_matrix_2d(vol_array);
            if (vol_mat.cols() != n_assets) {
                throw std::runtime_error("volatility_source matrix column count must match the number of asset columns.");
            }
            return population_sds(vol_mat);
        }
    }

    if (py::isinstance<py::sequence>(volatility_source) && !py::isinstance<py::str>(volatility_source)) {
        const std::vector<double> values = volatility_source.cast<std::vector<double>>();
        if (static_cast<Eigen::Index>(values.size()) != n_assets) {
            throw std::runtime_error("volatility_source vector length must match the number of asset columns.");
        }
        return Eigen::Map<const VectorXd>(values.data(), static_cast<Eigen::Index>(values.size()));
    }

    throw std::runtime_error("volatility_source must be a 1D volatility vector or a 2D raw-signal matrix.");
}

VectorXd normalized_signal_weights(std::size_t n_signals, const std::vector<double>& signal_weights) {
    if (n_signals == 0) {
        throw std::runtime_error("signal_input must contain at least one signal matrix.");
    }

    VectorXd weights(n_signals);
    if (signal_weights.empty()) {
        weights.setConstant(1.0 / static_cast<double>(n_signals));
        return weights;
    }

    if (signal_weights.size() != n_signals) {
        throw std::runtime_error("signal_weights length must match the number of signal matrices.");
    }

    double weight_sum = 0.0;
    for (std::size_t i = 0; i < n_signals; ++i) {
        const double value = signal_weights[i];
        if (!std::isfinite(value) || value < 0.0) {
            throw std::runtime_error("signal_weights must be finite and non-negative.");
        }
        weights(static_cast<Eigen::Index>(i)) = value;
        weight_sum += value;
    }

    if (weight_sum <= 0.0) {
        throw std::runtime_error("signal_weights must sum to a positive value.");
    }

    return weights / weight_sum;
}

MatrixXd gerber_cov_stat1(
    const MatrixXd& rets_mat,
    double threshold,
    const std::string& center_method,
    bool corr
) {
    if (!(threshold > 0.0 && threshold < 1.0)) {
        throw std::runtime_error("threshold must be between 0 and 1.");
    }

    const VectorXd sd_vec = population_sds(rets_mat);
    const MatrixXd adj_rets_mat = center_matrix(rets_mat, center_method);
    MatrixXd corr_mat = MatrixXd::Zero(rets_mat.cols(), rets_mat.cols());

    for (Eigen::Index i = 0; i < rets_mat.cols(); ++i) {
        for (Eigen::Index j = 0; j <= i; ++j) {
            Eigen::Index pos = 0;
            Eigen::Index neg = 0;
            Eigen::Index nn = 0;

            for (Eigen::Index row = 0; row < rets_mat.rows(); ++row) {
                const double x = adj_rets_mat(row, i);
                const double y = adj_rets_mat(row, j);
                const double x_threshold = threshold * sd_vec(i);
                const double y_threshold = threshold * sd_vec(j);

                if ((x >= x_threshold && y >= y_threshold) || (x <= -x_threshold && y <= -y_threshold)) {
                    ++pos;
                } else if ((x >= x_threshold && y <= -y_threshold) || (x <= -x_threshold && y >= y_threshold)) {
                    ++neg;
                } else if (std::abs(x) < x_threshold && std::abs(y) < y_threshold) {
                    ++nn;
                }
            }

            const double denominator = static_cast<double>(rets_mat.rows() - nn);
            corr_mat(i, j) = static_cast<double>(pos - neg) / denominator;
            corr_mat(j, i) = corr_mat(i, j);
        }
    }

    if (corr) {
        return corr_mat;
    }

    return corr_to_cov(corr_mat, sd_vec);
}

MatrixXd modified_gerber_corr(
    const MatrixXd& signal_mat,
    double gamma,
    double n,
    const std::optional<double>& half_life
) {
    MatrixXd corr_mat = MatrixXd::Zero(signal_mat.cols(), signal_mat.cols());

    for (Eigen::Index i = 0; i < signal_mat.cols(); ++i) {
        // The weighted ratio defines distinct-pair co-movement only.
        // Self-association is imposed as one below.
        for (Eigen::Index j = 0; j < i; ++j) {
            double numerator = 0.0;
            double denominator = 0.0;

            for (Eigen::Index row = 0; row < signal_mat.rows(); ++row) {
                const double x = signal_mat(row, i);
                const double y = signal_mat(row, j);
                const double weight = mgs_weight(x, y, gamma, n) * time_weight(half_life, signal_mat.rows(), row);
                numerator += sign_or_zero(x * y) * weight;
                denominator += weight;
            }

            corr_mat(i, j) = (denominator == 0.0) ? 0.0 : (numerator / denominator);
            corr_mat(j, i) = corr_mat(i, j);
        }
    }

    corr_mat.diagonal().setOnes();
    return corr_mat;
}

std::optional<double> parse_half_life(const py::object& half_life) {
    if (half_life.is_none()) {
        return std::nullopt;
    }
    const double value = half_life.cast<double>();
    if (!(value > 0.0) || !std::isfinite(value)) {
        throw std::runtime_error("half_life must be finite and positive.");
    }
    return value;
}

MatrixXd composite_modified_gerber_stat(
    const std::vector<MatrixXd>& signal_mats,
    double gamma,
    double n,
    bool corr,
    const std::optional<double>& half_life,
    const std::vector<double>& signal_weights,
    const std::optional<VectorXd>& volatility_stds
) {
    if (signal_mats.empty()) {
        throw std::runtime_error("signal_input must contain at least one signal matrix.");
    }

    const Eigen::Index n_periods = signal_mats.front().rows();
    const Eigen::Index n_assets = signal_mats.front().cols();
    for (std::size_t m = 1; m < signal_mats.size(); ++m) {
        if (signal_mats[m].rows() != n_periods || signal_mats[m].cols() != n_assets) {
            throw std::runtime_error("All signal matrices must have the same shape.");
        }
    }

    const VectorXd weights = normalized_signal_weights(signal_mats.size(), signal_weights);
    MatrixXd composite_corr = MatrixXd::Zero(n_assets, n_assets);

    for (std::size_t m = 0; m < signal_mats.size(); ++m) {
        composite_corr += weights(static_cast<Eigen::Index>(m)) * modified_gerber_corr(signal_mats[m], gamma, n, half_life);
    }
    composite_corr.diagonal().setOnes();

    if (corr) {
        return composite_corr;
    }

    const VectorXd sd_vec = volatility_stds.has_value() ? *volatility_stds : population_sds(signal_mats.front());
    if (sd_vec.size() != n_assets) {
        throw std::runtime_error("volatility_source must match the number of asset columns.");
    }
    return corr_to_cov(composite_corr, sd_vec);
}

}  // namespace


PYBIND11_MODULE(cgerber, m) {
    m.def(
        "c_gerber_cov_stat1",
        &gerber_cov_stat1,
        py::arg("rets_mat"),
        py::arg("threshold") = 0.5,
        py::arg("center_method") = "zero",
        py::arg("corr") = false,
        "Compute Gerber covariance statistic 1 from a return matrix."
    );

    m.def(
        "c_modified_gerber_stat",
        [](
            py::object signal_input,
            double gamma,
            double n,
            bool corr,
            py::object half_life,
            py::object signal_weights,
            py::object volatility_source
        ) {
            const std::vector<MatrixXd> signal_mats = parse_signal_input(signal_input);

            std::vector<double> weights;
            if (!signal_weights.is_none()) {
                weights = signal_weights.cast<std::vector<double>>();
            }

            std::optional<VectorXd> volatility_stds = std::nullopt;
            if (!volatility_source.is_none()) {
                volatility_stds = parse_volatility_stds(volatility_source, signal_mats.front().cols());
            }
            const std::optional<double> half_life_value = parse_half_life(half_life);

            return composite_modified_gerber_stat(
                signal_mats,
                gamma,
                n,
                corr,
                half_life_value,
                weights,
                volatility_stds
            );
        },
        py::arg("signal_input"),
        py::arg("gamma") = 1.0,
        py::arg("n") = 2.0,
        py::arg("corr") = false,
        py::arg("half_life") = py::none(),
        py::arg("signal_weights") = py::none(),
        py::arg("volatility_source") = py::none(),
        "Compute the paper-style Modified Gerber Statistic for one or more aligned signal matrices."
    );

}
