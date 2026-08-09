"""
Optera Category-Wise Demand Profiler & Distribution Fitting Engine

Performs quantitative demand modeling on daily aggregated demand time-series:
• Statistical profiling (Central tendency, volatility indicators, percentiles, higher moments)
• Confidence intervals for mean demand (e.g., 95%, 99%)
• Parametric distribution fitting (Normal, Poisson, Gamma, Negative Binomial, Empirical)
• Goodness-of-Fit testing (Log-Likelihood, AIC, BIC, Kolmogorov-Smirnov)
• Demand pattern & regime detection (Trend, Seasonality, Stationarity ADF, Croston Intermittency)
• Financial margin analysis
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import adfuller

logger = logging.getLogger("OpteraDemandProfiler")


class CategoryDemandProfiler:
    """
    Quantitative Demand Analytical Engine for Optera.
    """

    def __init__(
        self,
        data: Union[str, Path, pd.DataFrame],
        confidence_levels: Optional[List[float]] = None,
        distributions: Optional[List[str]] = None,
        goodness_of_fit_metric: str = "aic"
    ):
        """
        Parameters
        ----------
        data : Union[str, Path, pd.DataFrame]
            Path to CSV or pandas DataFrame containing daily category demand.
        confidence_levels : Optional[List[float]]
            List of confidence levels for mean demand intervals. Default: [0.95, 0.99]
        distributions : Optional[List[str]]
            List of distribution names to fit. Default: ["normal", "poisson", "gamma", "nbinom"]
        goodness_of_fit_metric : str
            Metric used to select the best distribution fit: "aic", "bic", or "ks"
        """
        if isinstance(data, (str, Path)):
            path = Path(data)
            self.df = pd.read_csv(path)
        elif isinstance(data, pd.DataFrame):
            self.df = data.copy()
        else:
            raise ValueError("Input must be a CSV file path or pandas DataFrame.")

        if "date" in self.df.columns:
            self.df["date"] = pd.to_datetime(self.df["date"])

        self.confidence_levels = confidence_levels if confidence_levels is not None else [0.95, 0.99]
        self.distributions = distributions if distributions is not None else ["normal", "poisson", "gamma", "nbinom"]
        self.gof_metric = goodness_of_fit_metric.lower()

    def profile_category(self, category_name: str) -> Dict[str, Any]:
        """
        Profiles a single product category time-series comprehensively.
        """
        cat_df = self.df[self.df["category"] == category_name].sort_values("date")
        if cat_df.empty:
            raise ValueError(f"Category '{category_name}' not found in dataset.")

        series = cat_df["quantity"].values
        n_obs = len(series)

        # -------------------------------------------------------------
        # 1. Statistical Profile & Central Tendency
        # -------------------------------------------------------------
        mean_val = float(np.mean(series))
        std_val = float(np.std(series, ddof=1)) if n_obs > 1 else 0.0
        var_val = float(np.var(series, ddof=1)) if n_obs > 1 else 0.0
        median_val = float(np.median(series))
        min_val = float(np.min(series))
        max_val = float(np.max(series))
        cv_val = float(std_val / mean_val) if mean_val > 0 else 0.0

        p25 = float(np.percentile(series, 25))
        p75 = float(np.percentile(series, 75))

        percentiles = {
            "p10": float(np.percentile(series, 10)),
            "p25": p25,
            "p50": median_val,
            "p75": p75,
            "p90": float(np.percentile(series, 90)),
            "p95": float(np.percentile(series, 95)),
            "p99": float(np.percentile(series, 99))
        }

        # Higher Moments
        skew_val = float(stats.skew(series)) if n_obs > 2 else 0.0
        kurt_val = float(stats.kurtosis(series)) if n_obs > 3 else 0.0

        # -------------------------------------------------------------
        # 2. Confidence Intervals for Mean Demand
        # -------------------------------------------------------------
        confidence_intervals = {}
        if n_obs > 1 and std_val > 0:
            sem = std_val / np.sqrt(n_obs)
            for clevel in self.confidence_levels:
                h_val = stats.t.ppf((1 + clevel) / 2.0, df=n_obs - 1) * sem
                confidence_intervals[f"{int(clevel * 100)}%"] = {
                    "lower": float(mean_val - h_val),
                    "upper": float(mean_val + h_val)
                }
        else:
            for clevel in self.confidence_levels:
                confidence_intervals[f"{int(clevel * 100)}%"] = {
                    "lower": mean_val,
                    "upper": mean_val
                }

        # -------------------------------------------------------------
        # 3. Parametric Distribution Fitting & Goodness of Fit (AIC/BIC)
        # -------------------------------------------------------------
        fitted_dists = {}
        int_data = np.round(series).astype(int)

        # A. Normal Distribution
        if "normal" in self.distributions:
            mu_n, std_n = mean_val, max(std_val, 1e-4)
            ll_norm = float(np.sum(stats.norm.logpdf(series, loc=mu_n, scale=std_n)))
            k_norm = 2
            aic_norm = 2 * k_norm - 2 * ll_norm
            bic_norm = k_norm * np.log(n_obs) - 2 * ll_norm
            ks_stat_n, ks_p_n = stats.kstest(series, "norm", args=(mu_n, std_n))
            fitted_dists["normal"] = {
                "name": "Normal (Gaussian)",
                "params": {"mu": float(mu_n), "sigma": float(std_n)},
                "log_likelihood": ll_norm,
                "aic": float(aic_norm),
                "bic": float(bic_norm),
                "ks_stat": float(ks_stat_n),
                "ks_pvalue": float(ks_p_n)
            }

        # B. Poisson Distribution
        if "poisson" in self.distributions:
            lam_p = max(mean_val, 1e-4)
            ll_poi = float(np.sum(stats.poisson.logpmf(int_data, mu=lam_p)))
            k_poi = 1
            aic_poi = 2 * k_poi - 2 * ll_poi
            bic_poi = k_poi * np.log(n_obs) - 2 * ll_poi
            fitted_dists["poisson"] = {
                "name": "Poisson",
                "params": {"lambda": float(lam_p)},
                "log_likelihood": ll_poi,
                "aic": float(aic_poi),
                "bic": float(bic_poi),
                "ks_stat": 0.0,
                "ks_pvalue": 0.0
            }

        # C. Gamma Distribution
        if "gamma" in self.distributions:
            try:
                shape_g, loc_g, scale_g = stats.gamma.fit(series, floc=0)
                ll_gam = float(np.sum(stats.gamma.logpdf(series, shape_g, loc=loc_g, scale=scale_g)))
                k_gam = 2
                aic_gam = 2 * k_gam - 2 * ll_gam
                bic_gam = k_gam * np.log(n_obs) - 2 * ll_gam
                ks_stat_g, ks_p_g = stats.kstest(series, "gamma", args=(shape_g, loc_g, scale_g))
                fitted_dists["gamma"] = {
                    "name": "Gamma",
                    "params": {"shape": float(shape_g), "scale": float(scale_g)},
                    "log_likelihood": ll_gam,
                    "aic": float(aic_gam),
                    "bic": float(bic_gam),
                    "ks_stat": float(ks_stat_g),
                    "ks_pvalue": float(ks_p_g)
                }
            except Exception as e:
                logger.warning("Gamma distribution fit failed for category %s: %s", category_name, e)

        # D. Negative Binomial Distribution
        if "nbinom" in self.distributions:
            try:
                if var_val > mean_val and mean_val > 0:
                    p_nb = mean_val / var_val
                    n_nb = (mean_val ** 2) / (var_val - mean_val)
                else:
                    p_nb = 0.95
                    n_nb = max(1.0, mean_val * 19.0)

                p_nb = float(np.clip(p_nb, 1e-4, 0.999))
                n_nb = float(max(0.1, n_nb))
                ll_nb = float(np.sum(stats.nbinom.logpmf(int_data, n=n_nb, p=p_nb)))
                k_nb = 2
                aic_nb = 2 * k_nb - 2 * ll_nb
                bic_nb = k_nb * np.log(n_obs) - 2 * ll_nb
                fitted_dists["nbinom"] = {
                    "name": "Negative Binomial",
                    "params": {"n": float(n_nb), "p": float(p_nb)},
                    "log_likelihood": ll_nb,
                    "aic": float(aic_nb),
                    "bic": float(bic_nb),
                    "ks_stat": 0.0,
                    "ks_pvalue": 0.0
                }
            except Exception as e:
                logger.warning("Negative Binomial fit failed for category %s: %s", category_name, e)

        # Determine best fitting distribution based on GOF metric (AIC / BIC)
        best_dist_key = "normal"
        best_gof_val = float("inf")
        for dkey, dinfo in fitted_dists.items():
            val = dinfo.get(self.gof_metric, dinfo.get("aic", float("inf")))
            if val < best_gof_val:
                best_gof_val = val
                best_dist_key = dkey

        best_fit_info = {
            "key": best_dist_key,
            "name": fitted_dists[best_dist_key]["name"],
            "metric": self.gof_metric.upper(),
            "score": best_gof_val,
            "details": fitted_dists[best_dist_key]
        }

        # -------------------------------------------------------------
        # 4. Demand Pattern & Regime Detection
        # -------------------------------------------------------------
        # A. Intermittency Classification (Croston / Syntetos-Boylan)
        non_zero_series = series[series > 0]
        n_non_zero = len(non_zero_series)
        adi = float(n_obs / n_non_zero) if n_non_zero > 0 else float("inf")
        if n_non_zero > 1:
            nz_mean = np.mean(non_zero_series)
            nz_std = np.std(non_zero_series, ddof=1)
            cv2 = float((nz_std / nz_mean) ** 2) if nz_mean > 0 else 0.0
        else:
            cv2 = 0.0

        if adi < 1.32 and cv2 < 0.49:
            intermittent_pattern = "Smooth"
        elif adi >= 1.32 and cv2 < 0.49:
            intermittent_pattern = "Intermittent"
        elif adi < 1.32 and cv2 >= 0.49:
            intermittent_pattern = "Erratic"
        else:
            intermittent_pattern = "Lumpy"

        # B. Stationarity Test (Augmented Dickey-Fuller)
        if n_obs >= 10:
            try:
                adf_res = adfuller(series, maxlag=None, regression="c", autolag="AIC")
                adf_stat, adf_p = float(adf_res[0]), float(adf_res[1])
                is_stationary = bool(adf_p < 0.05)
            except Exception as e:
                logger.warning("ADF test failed for category %s: %s", category_name, e)
                adf_stat, adf_p, is_stationary = 0.0, 1.0, False
        else:
            adf_stat, adf_p, is_stationary = 0.0, 1.0, False

        # C. Trend Detection (Linear Regression)
        time_idx = np.arange(n_obs)
        slope, intercept, r_value, p_value, std_err = stats.linregress(time_idx, series)
        if p_value < 0.05:
            trend_pattern = "Upward Trend" if slope > 0 else "Downward Trend"
        else:
            trend_pattern = "No Significant Trend"

        # D. Seasonality Detection (Autocorrelation at Lag 7)
        if n_obs >= 14:
            s_diff = series - mean_val
            c0 = float(np.sum(s_diff ** 2))
            if c0 > 0:
                autocorr_lag7 = float(np.sum(s_diff[:-7] * s_diff[7:]) / c0)
            else:
                autocorr_lag7 = 0.0
            seasonality_pattern = "Weekly Seasonality" if autocorr_lag7 > 0.15 else "No Strong Seasonality"
        else:
            autocorr_lag7 = 0.0
            seasonality_pattern = "Insufficient Data for Seasonality"

        # Combined Regime Description
        regime_summary = f"{intermittent_pattern} ({'Stationary' if is_stationary else 'Non-Stationary'}, {trend_pattern}, {seasonality_pattern})"

        # -------------------------------------------------------------
        # 5. Financial Metrics
        # -------------------------------------------------------------
        price_mean = float(cat_df["unit_price"].mean()) if "unit_price" in cat_df.columns else 0.0
        cost_mean = float(cat_df["unit_cost"].mean()) if "unit_cost" in cat_df.columns else 0.0
        margin_pct = float(((price_mean - cost_mean) / price_mean) * 100.0) if price_mean > 0 else 0.0
        tot_rev = float(cat_df["revenue"].sum()) if "revenue" in cat_df.columns else 0.0

        return {
            "category": category_name,
            "observations": n_obs,
            "total_quantity": float(np.sum(series)),
            "mean": mean_val,
            "median": median_val,
            "std": std_val,
            "variance": var_val,
            "cv": cv_val,
            "min": min_val,
            "max": max_val,
            "p25": p25,
            "p75": p75,
            "percentiles": percentiles,
            "skewness": skew_val,
            "kurtosis": kurt_val,
            "confidence_intervals": confidence_intervals,
            "fitted_distributions": fitted_dists,
            "best_fit": best_fit_info,
            "pattern_detection": {
                "intermittent_pattern": intermittent_pattern,
                "adi": adi,
                "cv2": cv2,
                "is_stationary": is_stationary,
                "adf_pvalue": adf_p,
                "trend": trend_pattern,
                "trend_slope": float(slope),
                "seasonality": seasonality_pattern,
                "autocorr_lag7": autocorr_lag7,
                "regime_summary": regime_summary
            },
            "financials": {
                "unit_price_mean": price_mean,
                "unit_cost_mean": cost_mean,
                "margin_pct": margin_pct,
                "total_revenue": tot_rev
            }
        }

    def profile_all_categories(self) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
        """
        Profiles all categories in the dataset and constructs a summary table.
        """
        if "category" not in self.df.columns:
            raise ValueError("Dataset missing 'category' column.")

        categories = sorted(list(self.df["category"].unique()))
        detailed_profiles = {}
        summary_rows = []

        for cat in categories:
            prof = self.profile_category(str(cat))
            detailed_profiles[str(cat)] = prof

            ci_95_str = f"[{prof['confidence_intervals']['95%']['lower']:.1f}, {prof['confidence_intervals']['95%']['upper']:.1f}]"

            summary_rows.append({
                "Category": cat,
                "Obs": prof["observations"],
                "Daily Mean": round(prof["mean"], 2),
                "Median": round(prof["median"], 2),
                "Daily Std": round(prof["std"], 2),
                "CV (Volatility)": round(prof["cv"], 2),
                "Skewness": round(prof["skewness"], 2),
                "Kurtosis": round(prof["kurtosis"], 2),
                "95% Mean CI": ci_95_str,
                "Best Distribution Fit": prof["best_fit"]["name"],
                "Demand Pattern": prof["pattern_detection"]["intermittent_pattern"],
                "Is Stationary": prof["pattern_detection"]["is_stationary"],
                "Trend": prof["pattern_detection"]["trend"],
                "Total Revenue ($)": round(prof["financials"]["total_revenue"], 2)
            })

        summary_df = pd.DataFrame(summary_rows).sort_values("Total Revenue ($)", ascending=False).reset_index(drop=True)
        return summary_df, detailed_profiles

    def compute_covariance_matrices(self) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
        """
        Computes N x N Category Demand Covariance Matrix (\Sigma) and Correlation Matrix (R).
        """
        if "date" not in self.df.columns or "category" not in self.df.columns or "quantity" not in self.df.columns:
            return {}, {}

        pivot_df = self.df.pivot_table(index="date", columns="category", values="quantity", aggfunc="sum").fillna(0.0)
        cov_df = pivot_df.cov()
        corr_df = pivot_df.corr()

        cov_dict = {}
        corr_dict = {}
        for c1 in cov_df.columns:
            c1_key = str(c1).lower().replace(" ", "_")
            cov_dict[c1_key] = {}
            corr_dict[c1_key] = {}
            for c2 in cov_df.columns:
                c2_key = str(c2).lower().replace(" ", "_")
                cov_dict[c1_key][c2_key] = round(float(cov_df.loc[c1, c2]), 4)
                corr_dict[c1_key][c2_key] = round(float(corr_df.loc[c1, c2]), 4)

        return cov_dict, corr_dict


def export_demand_model(
    detailed_profiles: Dict[str, Dict[str, Any]],
    output_path: Union[str, Path],
    cov_dict: Optional[Dict[str, Dict[str, float]]] = None,
    corr_dict: Optional[Dict[str, Dict[str, float]]] = None
) -> Path:
    """
    Constructs a structured JSON object representing the demand parameters of each category
    for downstream optimization (IPSO / PSO) models.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    categories = list(detailed_profiles.keys())
    cat_keys = [cat.lower().replace(" ", "_") for cat in categories]

    # Create root object containing per-category parameters + covariance matrix
    demand_model = {}

    if cov_dict:
        demand_model["metadata"] = {
            "covariance_matrix": cov_dict,
            "correlation_matrix": corr_dict if corr_dict else {}
        }

    for cat, prof in detailed_profiles.items():
        cat_key = cat.lower().replace(" ", "_")
        best_info = prof["best_fit"]
        best_dist_key = best_info["key"]
        best_dist_details = prof["fitted_distributions"].get(best_dist_key, {})

        ci_dict = {}
        for clevel, bounds in prof["confidence_intervals"].items():
            ci_dict[clevel] = [round(bounds["lower"], 2), round(bounds["upper"], 2)]

        cv_val = float(prof["cv"])
        is_stat = prof["pattern_detection"]["is_stationary"]
        pat_reg = prof["pattern_detection"]["intermittent_pattern"]

        cv_comp = min(1.0, cv_val / 0.5)
        stat_pen = 0.2 if not is_stat else 0.0
        int_pen = 0.2 if pat_reg in ["Intermittent", "Erratic", "Lumpy"] else 0.0
        risk_score = round(min(1.0, 0.6 * cv_comp + stat_pen + int_pen), 2)

        if risk_score < 0.25:
            risk_level = "Low"
        elif risk_score < 0.50:
            risk_level = "Moderate"
        elif risk_score < 0.75:
            risk_level = "High"
        else:
            risk_level = "Critical"

        demand_model[cat_key] = {
            "category": cat,
            "distribution": best_info["name"],
            "distribution_parameters": {
                k: round(float(v), 4) for k, v in best_dist_details.get("params", {}).items()
            },
            "gof_criterion": best_info["metric"],
            "gof_score": round(float(best_info["score"]), 2),
            "mean": round(float(prof["mean"]), 2),
            "median": round(float(prof["median"]), 2),
            "std": round(float(prof["std"]), 2),
            "variance": round(float(prof["variance"]), 2),
            "volatility_cv": round(cv_val, 4),
            "cv": round(cv_val, 4),
            "risk_profile": {
                "level": risk_level,
                "score": risk_score
            },
            "min": round(float(prof["min"]), 2),
            "max": round(float(prof["max"]), 2),
            "p25": round(float(prof["p25"]), 2),
            "p75": round(float(prof["p75"]), 2),
            "skewness": round(float(prof["skewness"]), 4),
            "kurtosis": round(float(prof["kurtosis"]), 4),
            "confidence_intervals": ci_dict,
            "pattern_regime": pat_reg,
            "is_stationary": is_stat,
            "trend": prof["pattern_detection"]["trend"],
            "seasonality": prof["pattern_detection"]["seasonality"],
            "financials": {
                "unit_price_mean": round(float(prof["financials"]["unit_price_mean"]), 2),
                "unit_cost_mean": round(float(prof["financials"]["unit_cost_mean"]), 2),
                "margin_pct": round(float(prof["financials"]["margin_pct"]), 2),
                "total_revenue": round(float(prof["financials"]["total_revenue"]), 2)
            }
        }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(demand_model, f, indent=2)

    logger.info("Exported Demand Model JSON -> %s", output_path)
    return output_path
