/**
 * Optera Monte Carlo Event-Driven Simulation Layer - Type Definitions (v2.0)
 *
 * Defines high-performance C++ data structures for category configurations,
 * inventory policies, day-by-day trial states, and aggregated simulation metrics.
 */

#ifndef OPTERA_SIMULATION_TYPES_HPP
#define OPTERA_SIMULATION_TYPES_HPP

#include <string>
#include <vector>
#include <map>
#include <cmath>
#include <algorithm>
#include <numeric>

namespace optera {

enum class DistributionType {
    NORMAL,
    POISSON,
    GAMMA,
    NEGATIVE_BINOMIAL
};

struct CategoryConfig {
    std::string name;
    DistributionType dist_type;
    double mean_demand;
    double std_demand;
    double param_alpha; // Gamma shape or NegBinom k
    double param_beta;  // Gamma scale or NegBinom p
    double unit_price;
    double unit_cost;
    double holding_cost_daily;
    double lead_time_days;
    double optimal_weight;
};

struct SimulationConfig {
    double total_budget = 500000.0;
    int num_simulations = 1000;
    int horizon_days = 90;
    double initial_cash = 100000.0;
    double min_cash_threshold = 50000.0;
    double safety_stock_z = 1.65; // 95% service level factor
};

struct PendingOrder {
    int arrival_day;
    double quantity;
    double cost_amount;
};

struct DailyCategoryState {
    double on_hand_inventory = 0.0;
    double pending_on_order = 0.0;
    double daily_demand = 0.0;
    double daily_sales = 0.0;
    double daily_stockout_qty = 0.0;
    double daily_revenue = 0.0;
    double daily_holding_cost = 0.0;
    double daily_lost_revenue = 0.0;
    double daily_lost_profit = 0.0;
    double reorder_point_s = 0.0;
    double reorder_quantity_q = 0.0;
    int reorder_trigger_count = 0;
};

struct DailyTrialRecord {
    int day;
    double cash_balance;
    double cumulative_profit;
    double total_daily_revenue;
    double total_daily_holding_cost;
    double total_daily_sales;
    double total_daily_demand;
    double total_daily_stockout_qty;
    std::map<std::string, DailyCategoryState> category_states;
};

struct TrialResult {
    int trial_id;
    double net_profit;
    double ending_cash;
    double total_revenue;
    double total_cogs;
    double total_holding_cost;
    double total_lost_revenue;
    double total_lost_profit;
    double overall_service_level; // Customer order probability
    double fill_rate;             // Total fulfilled demand / Total demand
    double inventory_turnover;    // COGS / Avg Inventory Value
    double days_of_inventory;     // Avg Inventory / Avg Daily Demand
    int total_reorder_events;
    std::map<std::string, double> category_stockout_prob;
    std::map<std::string, double> category_revenue;
    std::map<std::string, double> category_cost;
    std::map<std::string, double> category_holding_cost;
    std::map<std::string, double> category_profit;
    std::map<std::string, double> category_roi;
    std::map<std::string, double> category_avg_inventory;
    std::map<std::string, double> category_max_inventory;
    std::map<std::string, double> category_min_inventory;
    std::map<std::string, double> category_ending_inventory;
    std::vector<DailyTrialRecord> daily_records;
};

struct SimulationSummary {
    int num_simulations;
    int horizon_days;
    double total_budget;
    double budget_used;
    double budget_unused;
    double mean_profit;
    double median_profit;
    double std_profit;
    double var_5pct;    // Value at Risk 5% (5th percentile profit)
    double cvar_5pct;   // Conditional Value at Risk 5% (Expected Shortfall)
    double profit_95pct;
    double prob_of_loss;
    double cash_reserve_risk; // P(Ending Cash < min_cash_threshold)
    double overall_service_level;
    double fill_rate;
    double inventory_turnover;
    double days_of_inventory;
    double total_lost_revenue;
    double total_lost_profit;
    double average_reorder_count;
    double average_order_quantity;
    double average_lead_time_days;
    int total_supplier_orders;
    std::map<std::string, double> category_stockout_prob;
    std::map<std::string, double> category_financial_revenue;
    std::map<std::string, double> category_financial_cost;
    std::map<std::string, double> category_financial_holding_cost;
    std::map<std::string, double> category_financial_lost_sales_qty;
    std::map<std::string, double> category_financial_profit;
    std::map<std::string, double> category_financial_roi;
    std::map<std::string, double> category_avg_inventory;
    std::map<std::string, double> category_max_inventory;
    std::map<std::string, double> category_min_inventory;
    std::map<std::string, double> category_ending_inventory;
    double execution_time_ms;
};

} // namespace optera

#endif // OPTERA_SIMULATION_TYPES_HPP
