/**
 * Optera Inventory Policy Module - Continuous Review (s, Q) Policy (v2.0)
 *
 * Implements safety stock calculation s_i = mu_i * L_i + z * sigma_i * sqrt(L_i),
 * reorder quantity Q_i, and lead-time order inflow pipeline management.
 */

#ifndef OPTERA_INVENTORY_POLICY_HPP
#define OPTERA_INVENTORY_POLICY_HPP

#include "types.hpp"
#include <deque>

namespace optera {

class InventoryPolicy {
public:
    static double calculate_reorder_point(const CategoryConfig& config, double z) {
        double lead_days = std::max(1.0, config.lead_time_days);
        double lead_demand_mean = config.mean_demand * lead_days;
        double lead_demand_std = config.std_demand * std::sqrt(lead_days);
        double safety_stock = z * lead_demand_std;
        return std::ceil(lead_demand_mean + safety_stock);
    }

    static double calculate_reorder_quantity(const CategoryConfig& config, double total_budget) {
        // EOQ / Budget proportional allocation
        double lead_days = std::max(1.0, config.lead_time_days);
        double cycle_demand = config.mean_demand * (lead_days * 2.0);
        double min_units = std::ceil(cycle_demand);
        double budget_cap_units = std::floor((config.optimal_weight * total_budget) / std::max(1.0, config.unit_cost));
        return std::max(min_units, budget_cap_units);
    }
};

class OrderPipeline {
private:
    std::vector<PendingOrder> orders_;
public:
    void place_order(int current_day, double lead_days, double quantity, double cost) {
        PendingOrder order;
        order.arrival_day = current_day + static_cast<int>(std::round(lead_days));
        order.quantity = quantity;
        order.cost_amount = cost;
        orders_.push_back(order);
    }

    double process_arrivals(int current_day) {
        double arrived_qty = 0.0;
        auto it = orders_.begin();
        while (it != orders_.end()) {
            if (it->arrival_day <= current_day) {
                arrived_qty += it->quantity;
                it = orders_.erase(it);
            } else {
                ++it;
            }
        }
        return arrived_qty;
    }

    double get_pending_quantity() const {
        double pending = 0.0;
        for (const auto& ord : orders_) {
            pending += ord.quantity;
        }
        return pending;
    }

    void clear() {
        orders_.clear();
    }
};

} // namespace optera

#endif // OPTERA_INVENTORY_POLICY_HPP
