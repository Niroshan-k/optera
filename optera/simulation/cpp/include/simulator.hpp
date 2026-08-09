/**
 * Optera Monte Carlo Simulator Module (v2.0)
 *
 * Runs multithreaded day-by-day stochastic simulations over M trial paths,
 * computing financial metrics, service levels, fill rates, inventory turnover,
 * lost sales, VaR, CVaR, and category performance summaries.
 */

#ifndef OPTERA_SIMULATOR_HPP
#define OPTERA_SIMULATOR_HPP

#include "types.hpp"
#include "demand_generator.hpp"
#include "inventory_policy.hpp"
#include <chrono>
#include <iostream>
#include <memory>

namespace optera {

class MonteCarloSimulator {
private:
    std::vector<CategoryConfig> categories_;
    SimulationConfig config_;

public:
    MonteCarloSimulator(const std::vector<CategoryConfig>& categories, const SimulationConfig& config)
        : categories_(categories), config_(config) {}

    TrialResult run_single_trial(int trial_id, unsigned int seed) {
        std::mt19937 rng(seed);

        TrialResult trial;
        trial.trial_id = trial_id;

        // Create category demand generators & pipelines
        std::map<std::string, std::unique_ptr<IDemandGenerator>> generators;
        std::map<std::string, OrderPipeline> pipelines;
        std::map<std::string, DailyCategoryState> states;

        double initial_spent = 0.0;

        for (const auto& cat : categories_) {
            generators[cat.name] = DemandGeneratorFactory::create(cat);

            DailyCategoryState state;
            state.reorder_point_s = InventoryPolicy::calculate_reorder_point(cat, config_.safety_stock_z);
            state.reorder_quantity_q = InventoryPolicy::calculate_reorder_quantity(cat, config_.total_budget);

            // Exact Initial Inventory Allocation based on PSO Budget Weight (I_0 = floor(w_i* * B / c_i))
            double budget_for_cat = config_.total_budget * cat.optimal_weight;
            double initial_units = std::floor(budget_for_cat / std::max(1.0, cat.unit_cost));
            state.on_hand_inventory = initial_units;

            initial_spent += state.on_hand_inventory * cat.unit_cost;
            states[cat.name] = state;
        }

        double cash_balance = config_.initial_cash;
        double cumulative_profit = 0.0;
        double total_generated_demand = 0.0;
        double total_fulfilled_demand = 0.0;
        double total_customer_orders = 0.0;
        double total_satisfied_orders = 0.0;

        // Day-by-day stochastic loop
        for (int day = 1; day <= config_.horizon_days; ++day) {
            DailyTrialRecord day_rec;
            day_rec.day = day;
            day_rec.total_daily_revenue = 0.0;
            day_rec.total_daily_holding_cost = 0.0;
            day_rec.total_daily_sales = 0.0;
            day_rec.total_daily_demand = 0.0;
            day_rec.total_daily_stockout_qty = 0.0;

            for (const auto& cat : categories_) {
                auto& st = states[cat.name];
                auto& pipe = pipelines[cat.name];

                // 1. Process Order Arrivals from Lead Time Pipeline
                double arrived = pipe.process_arrivals(day);
                st.on_hand_inventory += arrived;
                st.pending_on_order = pipe.get_pending_quantity();

                // 2. Generate Stochastic Demand
                double demand = std::round(generators[cat.name]->sample(rng));
                st.daily_demand = demand;
                total_generated_demand += demand;
                total_customer_orders += (demand > 0 ? 1.0 : 0.0);

                // 3. Execute Sales & Stockouts
                double sales = std::min(st.on_hand_inventory, demand);
                double stockout_qty = demand - sales;

                st.daily_sales = sales;
                st.daily_stockout_qty = stockout_qty;
                st.on_hand_inventory -= sales;

                total_fulfilled_demand += sales;
                if (demand > 0 && stockout_qty == 0.0) {
                    total_satisfied_orders += 1.0;
                }

                // 4. Financial Calculations
                double revenue = sales * cat.unit_price;
                double holding_cost = st.on_hand_inventory * cat.holding_cost_daily;
                double lost_rev = stockout_qty * cat.unit_price;
                double margin = cat.unit_price - cat.unit_cost;
                double lost_profit = stockout_qty * margin;

                st.daily_revenue = revenue;
                st.daily_holding_cost = holding_cost;
                st.daily_lost_revenue = lost_rev;
                st.daily_lost_profit = lost_profit;

                day_rec.total_daily_revenue += revenue;
                day_rec.total_daily_holding_cost += holding_cost;
                day_rec.total_daily_sales += sales;
                day_rec.total_daily_demand += demand;
                day_rec.total_daily_stockout_qty += stockout_qty;

                // 5. Base-Stock (s, S) Reorder Trigger with MOQ, Working Capital, and Warehouse Capacity Constraints
                double inventory_position = st.on_hand_inventory + st.pending_on_order;
                if (inventory_position <= st.reorder_point_s) {
                    double review_cycle_days = 14.0;
                    double moq = 100.0;
                    double order_up_to_S = st.reorder_point_s + std::ceil(cat.mean_demand * review_cycle_days);
                    double max_warehouse_capacity = std::ceil(order_up_to_S * 2.5);

                    double q_needed = std::max(0.0, order_up_to_S - inventory_position);
                    double q_target = std::ceil(cat.mean_demand * review_cycle_days);
                    double q_affordable = std::floor(cash_balance / std::max(1.0, cat.unit_cost));
                    double q_storage_space = std::max(0.0, max_warehouse_capacity - inventory_position);

                    double q_to_order = std::min({q_needed, q_target, q_affordable, q_storage_space});

                    if (q_to_order >= moq) {
                        double order_cost = q_to_order * cat.unit_cost;
                        cash_balance -= order_cost;
                        pipe.place_order(day, cat.lead_time_days, q_to_order, order_cost);
                        st.reorder_trigger_count++;
                        st.pending_on_order = pipe.get_pending_quantity();
                    }
                }

                day_rec.category_states[cat.name] = st;
            }

            // Update Cash & Cumulative Profit
            cash_balance += day_rec.total_daily_revenue - day_rec.total_daily_holding_cost;
            double net_daily_profit = day_rec.total_daily_revenue - day_rec.total_daily_holding_cost;
            cumulative_profit += net_daily_profit;

            day_rec.cash_balance = cash_balance;
            day_rec.cumulative_profit = cumulative_profit;
            trial.daily_records.push_back(day_rec);
        }

        // Trial Summary Calculations
        trial.net_profit = cumulative_profit;
        trial.ending_cash = cash_balance;
        trial.total_revenue = 0.0;
        trial.total_holding_cost = 0.0;
        trial.total_cogs = 0.0;
        trial.total_lost_revenue = 0.0;
        trial.total_lost_profit = 0.0;
        trial.total_reorder_events = 0;

        trial.overall_service_level = (total_customer_orders > 0) ? (total_satisfied_orders / total_customer_orders) : 1.0;
        trial.fill_rate = (total_generated_demand > 0) ? (total_fulfilled_demand / total_generated_demand) : 1.0;

        double total_inventory_val_sum = 0.0;
        double total_daily_demand_sum = 0.0;

        for (const auto& cat : categories_) {
            double cat_rev = 0.0;
            double cat_hold = 0.0;
            double cat_sales = 0.0;
            double cat_demand = 0.0;
            double cat_stockouts = 0.0;
            double cat_lost_rev = 0.0;
            double cat_lost_prof = 0.0;
            double cat_inv_sum = 0.0;
            double cat_max_inv = 0.0;
            double cat_min_inv = 1e9;

            for (const auto& d_rec : trial.daily_records) {
                const auto& c_st = d_rec.category_states.at(cat.name);
                cat_rev += c_st.daily_revenue;
                cat_hold += c_st.daily_holding_cost;
                cat_sales += c_st.daily_sales;
                cat_demand += c_st.daily_demand;
                cat_stockouts += (c_st.daily_stockout_qty > 0 ? 1.0 : 0.0);
                cat_lost_rev += c_st.daily_lost_revenue;
                cat_lost_prof += c_st.daily_lost_profit;

                double inv = c_st.on_hand_inventory;
                cat_inv_sum += inv;
                cat_max_inv = std::max(cat_max_inv, inv);
                cat_min_inv = std::min(cat_min_inv, inv);
            }

            double cogs = cat_sales * cat.unit_cost;
            double net_prof = cat_rev - cat_hold - cogs;
            double roi = (cogs > 0) ? (net_prof / cogs) * 100.0 : 0.0;
            double avg_inv = cat_inv_sum / config_.horizon_days;
            double stockout_prob = cat_stockouts / config_.horizon_days;

            trial.category_revenue[cat.name] = cat_rev;
            trial.category_cost[cat.name] = cogs;
            trial.category_holding_cost[cat.name] = cat_hold;
            trial.category_profit[cat.name] = net_prof;
            trial.category_roi[cat.name] = roi;
            trial.category_stockout_prob[cat.name] = stockout_prob;
            trial.category_avg_inventory[cat.name] = avg_inv;
            trial.category_max_inventory[cat.name] = cat_max_inv;
            trial.category_min_inventory[cat.name] = cat_min_inv;
            trial.category_ending_inventory[cat.name] = states[cat.name].on_hand_inventory;

            trial.total_revenue += cat_rev;
            trial.total_holding_cost += cat_hold;
            trial.total_cogs += cogs;
            trial.total_lost_revenue += cat_lost_rev;
            trial.total_lost_profit += cat_lost_prof;
            trial.total_reorder_events += states[cat.name].reorder_trigger_count;

            total_inventory_val_sum += (avg_inv * cat.unit_cost);
            total_daily_demand_sum += (cat_demand / config_.horizon_days);
        }

        trial.inventory_turnover = (total_inventory_val_sum > 0) ? (trial.total_cogs / total_inventory_val_sum) : 0.0;
        trial.days_of_inventory = (total_daily_demand_sum > 0) ? ((total_inventory_val_sum / 50.0) / total_daily_demand_sum) : 0.0;

        return trial;
    }

    SimulationSummary run_simulation() {
        auto t0 = std::chrono::high_resolution_clock::now();

        std::vector<TrialResult> trials(config_.num_simulations);

        // Multithreaded loop over trial paths
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < config_.num_simulations; ++i) {
            unsigned int seed = 1337 + i * 997;
            trials[i] = run_single_trial(i + 1, seed);
        }

        // Aggregate Summary Statistics across all trials
        SimulationSummary summary;
        summary.num_simulations = config_.num_simulations;
        summary.horizon_days = config_.horizon_days;
        summary.total_budget = config_.total_budget;

        double sum_profit = 0.0;
        double sum_service = 0.0;
        double sum_fill = 0.0;
        double sum_turnover = 0.0;
        double sum_doi = 0.0;
        double sum_lost_rev = 0.0;
        double sum_lost_prof = 0.0;
        int cash_risk_count = 0;
        int loss_count = 0;

        std::vector<double> profit_vec;
        profit_vec.reserve(config_.num_simulations);

        for (const auto& tr : trials) {
            sum_profit += tr.net_profit;
            sum_service += tr.overall_service_level;
            sum_fill += tr.fill_rate;
            sum_turnover += tr.inventory_turnover;
            sum_doi += tr.days_of_inventory;
            sum_lost_rev += tr.total_lost_revenue;
            sum_lost_prof += tr.total_lost_profit;

            profit_vec.push_back(tr.net_profit);

            if (tr.net_profit < 0.0) loss_count++;
            if (tr.ending_cash < config_.min_cash_threshold) cash_risk_count++;
        }

        std::sort(profit_vec.begin(), profit_vec.end());

        summary.mean_profit = sum_profit / config_.num_simulations;
        int med_idx = config_.num_simulations / 2;
        summary.median_profit = profit_vec[med_idx];

        // Profit Std Dev
        double sq_err = 0.0;
        for (double p : profit_vec) {
            sq_err += (p - summary.mean_profit) * (p - summary.mean_profit);
        }
        summary.std_profit = std::sqrt(sq_err / config_.num_simulations);

        // VaR 5% (5th percentile)
        int var_idx = static_cast<int>(std::floor(0.05 * config_.num_simulations));
        summary.var_5pct = profit_vec[std::max(0, var_idx)];

        // CVaR 5% (Expected Shortfall - mean of worst 5% trials)
        double cvar_sum = 0.0;
        int cvar_count = std::max(1, var_idx);
        for (int i = 0; i < cvar_count; ++i) {
            cvar_sum += profit_vec[i];
        }
        summary.cvar_5pct = cvar_sum / cvar_count;

        // 95th Percentile Profit
        int p95_idx = static_cast<int>(std::floor(0.95 * config_.num_simulations));
        summary.profit_95pct = profit_vec[std::min(config_.num_simulations - 1, p95_idx)];

        summary.prob_of_loss = (static_cast<double>(loss_count) / config_.num_simulations) * 100.0;
        summary.cash_reserve_risk = (static_cast<double>(cash_risk_count) / config_.num_simulations) * 100.0;
        summary.overall_service_level = (sum_service / config_.num_simulations) * 100.0;
        summary.fill_rate = (sum_fill / config_.num_simulations) * 100.0;
        summary.inventory_turnover = sum_turnover / config_.num_simulations;
        summary.days_of_inventory = sum_doi / config_.num_simulations;
        summary.total_lost_revenue = sum_lost_rev / config_.num_simulations;
        summary.total_lost_profit = sum_lost_prof / config_.num_simulations;

        // Category-wise averages
        for (const auto& cat : categories_) {
            double c_stk = 0.0, c_rev = 0.0, c_cost = 0.0, c_hold = 0.0, c_prof = 0.0, c_roi = 0.0;
            double c_avg_inv = 0.0, c_max_inv = 0.0, c_min_inv = 0.0, c_end_inv = 0.0;

            for (const auto& tr : trials) {
                c_stk += tr.category_stockout_prob.at(cat.name);
                c_rev += tr.category_revenue.at(cat.name);
                c_cost += tr.category_cost.at(cat.name);
                c_hold += tr.category_holding_cost.at(cat.name);
                c_prof += tr.category_profit.at(cat.name);
                c_roi += tr.category_roi.at(cat.name);
                c_avg_inv += tr.category_avg_inventory.at(cat.name);
                c_max_inv += tr.category_max_inventory.at(cat.name);
                c_min_inv += tr.category_min_inventory.at(cat.name);
                c_end_inv += tr.category_ending_inventory.at(cat.name);
            }

            double N = static_cast<double>(config_.num_simulations);
            summary.category_stockout_prob[cat.name] = (c_stk / N) * 100.0;
            summary.category_financial_revenue[cat.name] = c_rev / N;
            summary.category_financial_cost[cat.name] = c_cost / N;
            summary.category_financial_holding_cost[cat.name] = c_hold / N;
            summary.category_financial_profit[cat.name] = c_prof / N;
            summary.category_financial_roi[cat.name] = c_roi / N;
            summary.category_avg_inventory[cat.name] = c_avg_inv / N;
            summary.category_max_inventory[cat.name] = c_max_inv / N;
            summary.category_min_inventory[cat.name] = c_min_inv / N;
            summary.category_ending_inventory[cat.name] = c_end_inv / N;
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        summary.execution_time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        return summary;
    }
};

} // namespace optera

#endif // OPTERA_SIMULATOR_HPP
