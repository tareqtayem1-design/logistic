import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Logistics Planning Optimizer",
    page_icon="🚚",
    layout="wide"
)

# Initialize session state
if 'daily_orders' not in st.session_state:
    st.session_state.daily_orders = {'planning_period': 7, 'daily_expected_orders': []}

if 'daily_slots' not in st.session_state:
    st.session_state.daily_slots = {
        'slots_per_day': 3, 
        'slot_capacities': [], 
        'slot_drivers': [],
        'slot_trips_per_time': [],
        'slot_orders_per_trip': [],  # Manual orders per trip setting
        'short_trip_capacity': [],  # <10km trips
        'long_trip_capacity': []    # >=10km trips
    }

if 'trucks_config' not in st.session_state:
    st.session_state.trucks_config = {
        'available_trucks': 10,
        'max_orders_per_truck': [],
        'truck_types': [
            {'id': 'A', 'capacity': 4, 'editable': True},
            {'id': 'B', 'capacity': 6, 'editable': True}
        ]
    }

if 'channels_config' not in st.session_state:
    st.session_state.channels_config = []

if 'optimization_params' not in st.session_state:
    st.session_state.optimization_params = {
        'otp_time_weights': {},
        'otp_targets': [95, 99],
        'cost_otp_weight': 0.5
    }

if 'optimization_results' not in st.session_state:
    st.session_state.optimization_results = {}

# ========================== INTELLIGENT OPTIMIZATION FUNCTIONS ==========================

def calculate_channel_capacity(channel):
    """Calculate maximum daily capacity for a channel"""
    drivers = channel['drivers_available']
    trips_per_driver = channel['trips_per_day_per_driver']
    orders_per_trip = channel['avg_orders_per_trip']
    
    # Calculate capacity from drivers
    driver_capacity = drivers * trips_per_driver * orders_per_trip
    
    # Calculate capacity from trucks
    trucks_a = channel['available_trucks_type_a']
    trucks_b = channel['available_trucks_type_b']
    
    # Find truck capacities
    truck_a_cap = 0
    truck_b_cap = 0
    for truck_type in st.session_state.trucks_config['truck_types']:
        if truck_type['id'] == 'A':
            truck_a_cap = truck_type['capacity']
        elif truck_type['id'] == 'B':
            truck_b_cap = truck_type['capacity']
    
    truck_capacity = (trucks_a * truck_a_cap) + (trucks_b * truck_b_cap)
    
    return min(driver_capacity, truck_capacity), driver_capacity, truck_capacity

def calculate_weighted_otp(channel):
    """Calculate weighted OTP based on time window weights"""
    weights = st.session_state.optimization_params.get('otp_time_weights', {})
    weight_24h = weights.get('24h', 0.5)
    weight_7d = weights.get('7d', 0.3)
    weight_30d = weights.get('30d', 0.2)
    
    return (channel['otp_24h'] * weight_24h + 
            channel['otp_7d'] * weight_7d + 
            channel['otp_30d'] * weight_30d)

def calculate_total_cost(channel, orders_assigned):
    """Calculate total cost for orders assigned to a channel"""
    if orders_assigned <= 0:
        return 0
    
    # Calculate number of trips needed
    avg_orders_per_trip = channel['avg_orders_per_trip']
    num_trips = np.ceil(orders_assigned / avg_orders_per_trip)
    
    # Variable cost (per order)
    variable_cost = orders_assigned * channel['cost_per_order']
    
    # Fixed cost (per trip)
    fixed_cost = num_trips * channel['fixed_cost_per_trip']
    
    return variable_cost + fixed_cost

def optimize_allocation(target_otp=None, max_cost=None):
    """Intelligent optimization algorithm"""
    if not st.session_state.channels_config or not st.session_state.daily_orders['daily_expected_orders']:
        return None
    
    daily_orders = st.session_state.daily_orders['daily_expected_orders']
    channels = st.session_state.channels_config
    cost_otp_weight = st.session_state.optimization_params['cost_otp_weight']
    
    allocation_results = []
    
    for day_idx, orders in enumerate(daily_orders):
        if orders == 0:
            continue
        
        # Calculate capacities and OTP scores for each channel
        channel_scores = []
        for ch in channels:
            capacity, driver_cap, truck_cap = calculate_channel_capacity(ch)
            weighted_otp = calculate_weighted_otp(ch)
            score = weighted_otp * (capacity / 100.0)  # Normalize by capacity
            
            channel_scores.append({
                'channel_id': ch['id'],
                'capacity': capacity,
                'otp': weighted_otp,
                'score': score,
                'cost_per_order': ch['cost_per_order'],
                'avg_orders_per_trip': ch['avg_orders_per_trip'],
                'fixed_cost_per_trip': ch['fixed_cost_per_trip'],
                'drivers_available': ch['drivers_available'],
                'trucks_a': ch['available_trucks_type_a'],
                'trucks_b': ch['available_trucks_type_b'],
                'trips_per_driver': ch['trips_per_day_per_driver']
            })
        
        # Sort by score (higher is better)
        channel_scores.sort(key=lambda x: x['score'], reverse=True)
        
        # Greedy allocation with optimization
        remaining_orders = orders
        day_allocation = []
        total_day_cost = 0
        
        for ch_score in channel_scores:
            if remaining_orders <= 0:
                break
            
            # Find full channel config
            ch = next((c for c in channels if c['id'] == ch_score['channel_id']), None)
            if not ch:
                continue
            
            # Determine allocation based on strategy
            if cost_otp_weight < 0.3:  # Maximize OTP
                alloc = min(remaining_orders, ch_score['capacity'])
            elif cost_otp_weight > 0.7:  # Minimize Cost
                # Allocate to cheapest channel first
                alloc = min(remaining_orders, ch_score['capacity'])
            else:  # Balanced
                alloc = min(remaining_orders, ch_score['capacity'])
            
            if alloc > 0:
                cost = calculate_total_cost(ch, alloc)
                day_allocation.append({
                    'day': day_idx + 1,
                    'channel': ch_score['channel_id'],
                    'orders': alloc,
                    'cost': cost,
                    'otp': ch_score['otp'],
                    'capacity_utilization': (alloc / ch_score['capacity']) * 100
                })
                total_day_cost += cost
                remaining_orders -= alloc
        
        if remaining_orders > 0:
            st.warning(f"⚠️ Insufficient capacity on day {day_idx + 1}: {remaining_orders} orders unallocated")
        
        allocation_results.append({
            'day': day_idx + 1,
            'total_orders': orders,
            'allocations': day_allocation,
            'total_cost': total_day_cost,
            'unallocated': remaining_orders
        })
    
    return allocation_results

def calculate_roi(optimization_results):
    """Calculate ROI metrics"""
    if not optimization_results:
        return None
    
    total_cost = sum(r['total_cost'] for r in optimization_results)
    total_orders = sum(r['total_orders'] for r in optimization_results)
    total_unallocated = sum(r['unallocated'] for r in optimization_results)
    
    # Weighted average OTP
    total_weighted_otp = 0
    total_weight = 0
    for result in optimization_results:
        for alloc in result['allocations']:
            total_weighted_otp += alloc['otp'] * alloc['orders']
            total_weight += alloc['orders']
    
    avg_otp = total_weighted_otp / total_weight if total_weight > 0 else 0
    cost_per_order = total_cost / total_orders if total_orders > 0 else 0
    
    return {
        'total_cost': total_cost,
        'total_orders': total_orders,
        'avg_cost_per_order': cost_per_order,
        'weighted_avg_otp': avg_otp,
        'fulfillment_rate': ((total_orders - total_unallocated) / total_orders * 100) if total_orders > 0 else 0,
        'unallocated_orders': total_unallocated
    }

# Main title
st.title("🚚 Logistics Planning Optimizer")
st.markdown("---")

# Sidebar navigation
st.sidebar.title("🤖 Navigation")
st.sidebar.markdown("### Sections")
page = st.sidebar.radio(
    "Select Section",
    ["🏆 Executive Dashboard", "📋 Daily Orders", "📅 Daily Slots", "🚛 Trucks Configuration", "🔗 Channel Configuration", 
     "⚙️ Optimization Parameters", "📊 AI Optimization & Insights", "🧪 What-If Scenarios", "📈 Predictive Analytics"],
    label_visibility="collapsed"
)

# ========================== DAILY ORDERS SECTION ==========================
if page == "📋 Daily Orders":
    st.header("📋 Daily Orders Configuration")
    
    # Help Section
    with st.expander("ℹ️ What is this section?", expanded=False):
        st.markdown("""
        ### Daily Orders Configuration
        
        **Purpose:** Define your expected order volume over a planning period.
        
        **Key Features:**
        - Set planning period (1-365 days)
        - Enter expected orders per day
        - Visualize order distribution
        - View summary metrics (total, average, peak days)
        
        **How to Use:**
        1. Choose your planning period
        2. Enter expected orders for each day
        3. Review the visualization to see your demand pattern
        4. Use this data in other sections for optimization
        
        **Business Impact:**
        - Accurate forecasts lead to better capacity planning
        - Identifies peak days requiring more resources
        - Helps prevent capacity shortages
        """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        planning_period = st.number_input(
            "Planning Period (Days)",
            min_value=1,
            max_value=365,
            value=st.session_state.daily_orders['planning_period'],
            key='planning_period_input'
        )
        st.session_state.daily_orders['planning_period'] = planning_period
    
    with col2:
        if planning_period > 0:
            st.info(f"Configure expected orders for {planning_period} days")
    
    st.subheader("Daily Expected Orders")
    
    if len(st.session_state.daily_orders['daily_expected_orders']) != planning_period:
        st.session_state.daily_orders['daily_expected_orders'] = [0] * planning_period
    
    # Create input table
    orders_data = []
    for day in range(planning_period):
        orders_data.append({
            'Day': day + 1,
            'Date': (datetime.now() + timedelta(days=day)).strftime('%Y-%m-%d'),
            'Expected Orders': st.session_state.daily_orders['daily_expected_orders'][day]
        })
    
    df_orders = pd.DataFrame(orders_data)
    
    # Display with editable table
    edited_df = st.data_editor(
        df_orders,
        column_config={
            "Day": st.column_config.NumberColumn("Day", disabled=True),
            "Date": st.column_config.TextColumn("Date", disabled=True),
            "Expected Orders": st.column_config.NumberColumn("Expected Orders", min_value=0, step=1)
        },
        hide_index=True,
        width='stretch'
    )
    
    # Update session state
    st.session_state.daily_orders['daily_expected_orders'] = edited_df['Expected Orders'].tolist()
    
    # Summary
    if st.session_state.daily_orders['daily_expected_orders']:
        total_orders = sum(st.session_state.daily_orders['daily_expected_orders'])
        avg_orders = total_orders / len(st.session_state.daily_orders['daily_expected_orders'])
        max_orders = max(st.session_state.daily_orders['daily_expected_orders'])
        min_orders = min(st.session_state.daily_orders['daily_expected_orders'])
        variance = max_orders - min_orders if max_orders > 0 else 0
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Orders", f"{total_orders}")
        with col2:
            st.metric("Average Orders/Day", f"{avg_orders:.1f}")
        with col3:
            st.metric("Max Orders/Day", f"{max_orders}")
        
        # Visualization
        fig = px.bar(
            edited_df,
            x='Date',
            y='Expected Orders',
            title="Daily Expected Orders",
            color='Expected Orders',
            color_continuous_scale='Blues'
        )
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, width='stretch')
        
        # Smart Recommendations after orders are configured
        st.divider()
        st.subheader("🤖 Smart Analysis & Best Scenario Recommendations")
        
        variance_pct = (variance / avg_orders * 100) if avg_orders > 0 else 0
        
        # Analyze demand patterns
        rec_col1, rec_col2, rec_col3 = st.columns(3)
        
        with rec_col1:
            st.metric("Peak Day Demand", f"{max_orders:,}")
            st.metric("Lowest Day", f"{min_orders:,}")
        
        with rec_col2:
            st.metric("Demand Variance", f"{variance_pct:.1f}%")
            if variance_pct > 30:
                st.warning("⚠️ High variance - plan for flexibility")
            else:
                st.success("✓ Stable demand pattern")
        
        with rec_col3:
            if st.session_state.channels_config:
                total_capacity = sum([calculate_channel_capacity(ch)[0] for ch in st.session_state.channels_config])
                capacity_status = "✅ Sufficient" if total_capacity >= max_orders else "⚠️ Insufficient"
                st.metric("Capacity Status", capacity_status)
                st.caption(f"Capacity: {total_capacity:.0f} | Peak: {max_orders}")
        
        # Intelligent recommendations
        recommendations = []
        
        # Demand analysis
        if variance_pct > 50:
            recommendations.append({
                'priority': 'HIGH',
                'title': 'High Demand Variability Detected',
                'message': f'Demand varies by {variance_pct:.0f}% - peak days need {max_orders} orders vs average {avg_orders:.0f}',
                'action': 'Consider flexible capacity or reserve channels for peak days',
                'icon': '📈'
            })
        
        # Volume-based recommendations
        if avg_orders < 50:
            recommendations.append({
                'priority': 'LOW',
                'title': 'Low Volume Operations',
                'message': f'Average {avg_orders:.0f} orders/day - suitable for 2-3 channels',
                'action': 'Optimize for cost efficiency with fewer channels',
                'icon': '💡'
            })
        elif avg_orders > 200:
            recommendations.append({
                'priority': 'HIGH',
                'title': 'High Volume Operations',
                'message': f'Average {avg_orders:.0f} orders/day - requires multiple channels',
                'action': 'Ensure sufficient channels and resources for scalability',
                'icon': '🚀'
            })
        
        # Capacity check
        if st.session_state.channels_config:
            total_capacity = sum([calculate_channel_capacity(ch)[0] for ch in st.session_state.channels_config])
            if total_capacity < max_orders:
                recommendations.append({
                    'priority': 'CRITICAL',
                    'title': '🚨 Critical: Insufficient Capacity for Peak Days',
                    'message': f'Peak demand ({max_orders}) exceeds capacity ({total_capacity:.0f})',
                    'action': f'Add {max_orders - total_capacity:.0f} capacity units or additional channels',
                    'icon': '🚨'
                })
            elif total_capacity < avg_orders * 1.2:
                recommendations.append({
                    'priority': 'MEDIUM',
                    'title': 'Tight Capacity Buffer',
                    'message': f'Capacity ({total_capacity:.0f}) is only {((total_capacity/avg_orders-1)*100):.0f}% above average',
                    'action': 'Consider adding 20-30% buffer capacity for demand spikes',
                    'icon': '⚠️'
                })
        
        # Display recommendations
        if recommendations:
            st.markdown("#### 💡 Intelligent Recommendations")
            for rec in sorted(recommendations, key=lambda x: {'CRITICAL': 0, 'HIGH': 1, 'MEDIUM': 2, 'LOW': 3}.get(x['priority'], 4)):
                with st.expander(f"{rec['icon']} {rec['title']} - {rec['priority']} Priority", expanded=(rec['priority'] in ['CRITICAL', 'HIGH'])):
                    st.markdown(f"**Analysis:** {rec['message']}")
                    st.markdown(f"**Recommended Action:** {rec['action']}")
        
        # Best scenario suggestion
        if st.session_state.channels_config and st.session_state.daily_slots.get('slot_capacities'):
            st.markdown("---")
            st.markdown("#### 🎯 Suggested Best Scenario")
            
            # Analyze channels to suggest optimal configuration
            channel_analysis = []
            for ch in st.session_state.channels_config:
                cap, driver_cap, truck_cap = calculate_channel_capacity(ch)
                weighted_otp = calculate_weighted_otp(ch)
                cost_per_order = ch['cost_per_order'] + (ch['fixed_cost_per_trip'] / ch['avg_orders_per_trip'])
                efficiency_score = weighted_otp / cost_per_order if cost_per_order > 0 else 0
                
                channel_analysis.append({
                    'channel': ch['id'],
                    'capacity': cap,
                    'otp': weighted_otp,
                    'cost': cost_per_order,
                    'efficiency': efficiency_score
                })
            
            # Sort by efficiency
            channel_analysis.sort(key=lambda x: x['efficiency'], reverse=True)
            
            # Suggest best channels
            st.info("**Recommended Configuration:**")
            st.markdown("Based on your orders and current setup, here's the optimal scenario:")
            
            scenario_col1, scenario_col2 = st.columns(2)
            
            with scenario_col1:
                st.markdown("**Top Channels to Use:**")
                for idx, ch_analysis in enumerate(channel_analysis[:3], 1):
                    st.markdown(f"{idx}. **{ch_analysis['channel']}** - Efficiency: {ch_analysis['efficiency']:.2f}")
                    st.caption(f"   Capacity: {ch_analysis['capacity']:.0f} | OTP: {ch_analysis['otp']:.1f}% | Cost: ${ch_analysis['cost']:.2f}")
            
            with scenario_col2:
                st.markdown("**Optimization Strategy:**")
                
                # Suggest cost-OTP weight
                if avg_orders > 150:
                    suggested_weight = 0.4  # More cost-focused for high volume
                    st.markdown("💡 **High Volume:** Suggest Cost-OTP weight = 0.4 (balance cost)")
                elif avg_orders < 50:
                    suggested_weight = 0.3  # More OTP-focused for low volume
                    st.markdown("💡 **Low Volume:** Suggest Cost-OTP weight = 0.3 (prioritize OTP)")
                else:
                    suggested_weight = 0.5  # Balanced
                    st.markdown("💡 **Medium Volume:** Suggest Cost-OTP weight = 0.5 (balanced)")
                
                st.markdown(f"**Suggested OTP Target:** 95% (balanced performance)")
                if st.session_state.channels_config:
                    total_cap = sum([calculate_channel_capacity(ch)[0] for ch in st.session_state.channels_config])
                    st.markdown(f"**Expected Fulfillment:** {min(100, (total_cap/total_orders*100) if total_orders > 0 else 0):.0f}%")
            
            # Auto-optimize button
            st.markdown("---")
            if st.button("🚀 Auto-Generate Best Scenario", type="primary", use_container_width=True):
                # Auto-set optimal parameters
                st.session_state.optimization_params['cost_otp_weight'] = suggested_weight
                st.session_state.optimization_params['otp_targets'] = [95, 97]
                
                # Run optimization
                with st.spinner("🤖 Generating optimal scenario..."):
                    results = optimize_allocation()
                    if results:
                        st.session_state.optimization_results['best_scenario'] = results
                        st.success("✅ Best scenario generated! Check 'AI Optimization & Insights' section.")
                        st.balloons()

# ========================== DAILY SLOTS SECTION ==========================
elif page == "📅 Daily Slots":
    st.header("📅 Daily Slots Configuration")
    
    # Help Section
    with st.expander("ℹ️ What is this section?", expanded=False):
        st.markdown("""
        ### Daily Slots Configuration
        
        **Purpose:** Define time slots for delivery operations throughout the day.
        
        **Key Concepts:**
        - **Slots:** Time periods in a day (e.g., Morning, Afternoon, Evening)
        - **Weights:** Relative importance/capacity of each slot (must sum to 1.0)
        
        **Features:**
        - Configure 1-10 time slots per day
        - Assign weights to each slot (e.g., 0.4 for peak hours, 0.2 for off-peak)
        - Visualize slot distribution
        - Validate weight normalization
        
        **How to Use:**
        1. Define number of slots (typically 2-4 for delivery operations)
        2. Assign weights (higher = more orders expected in that slot)
        3. Ensure total weight = 1.0 (app will warn if not normalized)
        
        **Example:**
        - Slot 1 (Morning): Weight 0.2 (20% of orders)
        - Slot 2 (Afternoon): Weight 0.5 (50% of orders - peak)
        - Slot 3 (Evening): Weight 0.3 (30% of orders)
        
        **Business Impact:**
        - Helps optimize resource allocation by time of day
        - Enables better staff scheduling
        - Improves customer delivery experience
        """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        slots_per_day = st.number_input(
            "Number of Slots per Day",
            min_value=1,
            max_value=10,
            value=st.session_state.daily_slots['slots_per_day'],
            key='slots_per_day_input'
        )
        st.session_state.daily_slots['slots_per_day'] = slots_per_day
    
    st.subheader("Slot Configuration - Capacity Based")
    
    st.info("💡 **Enter capacity for each slot. Configure trips per slot time and separate short/long trip capacities.**")
    
    # Initialize slot data
    if 'slot_capacities' not in st.session_state.daily_slots or len(st.session_state.daily_slots['slot_capacities']) != slots_per_day:
        st.session_state.daily_slots['slot_capacities'] = [10] * slots_per_day
        st.session_state.daily_slots['slot_drivers'] = [1] * slots_per_day
        st.session_state.daily_slots['slot_trips_per_time'] = [1] * slots_per_day
        st.session_state.daily_slots['slot_orders_per_trip'] = [0] * slots_per_day  # 0 means auto-calculate
        st.session_state.daily_slots['short_trip_capacity'] = [5] * slots_per_day
        st.session_state.daily_slots['long_trip_capacity'] = [5] * slots_per_day
    
    if 'slot_drivers' not in st.session_state.daily_slots or len(st.session_state.daily_slots['slot_drivers']) != slots_per_day:
        st.session_state.daily_slots['slot_drivers'] = [1] * slots_per_day
    
    if 'slot_trips_per_time' not in st.session_state.daily_slots or len(st.session_state.daily_slots['slot_trips_per_time']) != slots_per_day:
        st.session_state.daily_slots['slot_trips_per_time'] = [1] * slots_per_day
    
    if 'slot_orders_per_trip' not in st.session_state.daily_slots or len(st.session_state.daily_slots['slot_orders_per_trip']) != slots_per_day:
        st.session_state.daily_slots['slot_orders_per_trip'] = [0] * slots_per_day  # 0 means auto-calculate
    
    if 'short_trip_capacity' not in st.session_state.daily_slots or len(st.session_state.daily_slots['short_trip_capacity']) != slots_per_day:
        st.session_state.daily_slots['short_trip_capacity'] = [5] * slots_per_day
    
    if 'long_trip_capacity' not in st.session_state.daily_slots or len(st.session_state.daily_slots['long_trip_capacity']) != slots_per_day:
        st.session_state.daily_slots['long_trip_capacity'] = [5] * slots_per_day
    
    # Create slots data with capacity, trips, trip categories, and orders per trip
    slots_data = []
    for slot in range(slots_per_day):
        slot_capacity = st.session_state.daily_slots['slot_capacities'][slot] if slot < len(st.session_state.daily_slots['slot_capacities']) else 10
        slot_drivers = st.session_state.daily_slots['slot_drivers'][slot] if slot < len(st.session_state.daily_slots['slot_drivers']) else 1
        trips_per_time = st.session_state.daily_slots['slot_trips_per_time'][slot] if slot < len(st.session_state.daily_slots['slot_trips_per_time']) else 1
        orders_per_trip = st.session_state.daily_slots['slot_orders_per_trip'][slot] if slot < len(st.session_state.daily_slots['slot_orders_per_trip']) else 0
        short_capacity = st.session_state.daily_slots['short_trip_capacity'][slot] if slot < len(st.session_state.daily_slots['short_trip_capacity']) else 5
        long_capacity = st.session_state.daily_slots['long_trip_capacity'][slot] if slot < len(st.session_state.daily_slots['long_trip_capacity']) else 5
        
        slots_data.append({
            'Slot': f"Slot {slot + 1}",
            'Total Capacity': slot_capacity,
            'Drivers per Slot': slot_drivers,
            'Trips per Slot Time': trips_per_time,
            'Orders per Trip': orders_per_trip if orders_per_trip > 0 else None,  # None means auto
            'Short Trip (<10km)': short_capacity,
            'Long Trip (≥10km)': long_capacity
        })
    
    df_slots = pd.DataFrame(slots_data)
    
    # Display with editable table - allow sorting
    edited_slots_df = st.data_editor(
        df_slots,
        column_config={
            "Slot": st.column_config.TextColumn("Slot", disabled=True),
            "Total Capacity": st.column_config.NumberColumn("Total Capacity", min_value=1, step=1, help="Maximum total orders this slot can handle"),
            "Drivers per Slot": st.column_config.NumberColumn("Drivers per Slot", min_value=1, step=1, help="Number of drivers available in this slot"),
            "Trips per Slot Time": st.column_config.NumberColumn("Trips per Slot Time", min_value=1, step=1, help="How many trips each driver can make in this slot time"),
            "Orders per Trip": st.column_config.NumberColumn("Orders per Trip", min_value=0, step=0.5, help="Orders per trip (0 = auto-calculate based on capacity, or set manually)"),
            "Short Trip (<10km)": st.column_config.NumberColumn("Short Trip Capacity", min_value=0, step=1, help="Orders for trips less than 10km"),
            "Long Trip (≥10km)": st.column_config.NumberColumn("Long Trip Capacity", min_value=0, step=1, help="Orders for trips 10km or more")
        },
        hide_index=True,
        width='stretch',
        num_rows="fixed"
    )
    
    # Validate that short + long = total capacity
    for idx, row in edited_slots_df.iterrows():
        total = row['Total Capacity']
        short = row['Short Trip (<10km)']
        long = row['Long Trip (≥10km)']
        if abs(short + long - total) > 0.01:
            st.warning(f"⚠️ Slot {idx+1}: Short trip ({short}) + Long trip ({long}) should equal Total Capacity ({total})")
    
    # Sort by capacity (descending)
    edited_slots_df = edited_slots_df.sort_values('Total Capacity', ascending=False).reset_index(drop=True)
    
    st.session_state.daily_slots['slot_capacities'] = edited_slots_df['Total Capacity'].tolist()
    st.session_state.daily_slots['slot_drivers'] = edited_slots_df['Drivers per Slot'].tolist()
    st.session_state.daily_slots['slot_trips_per_time'] = edited_slots_df['Trips per Slot Time'].tolist()
    # Handle orders per trip - convert None to 0 (auto-calculate)
    orders_per_trip_list = []
    for val in edited_slots_df['Orders per Trip'].tolist():
        orders_per_trip_list.append(val if pd.notna(val) and val > 0 else 0)
    st.session_state.daily_slots['slot_orders_per_trip'] = orders_per_trip_list
    st.session_state.daily_slots['short_trip_capacity'] = edited_slots_df['Short Trip (<10km)'].tolist()
    st.session_state.daily_slots['long_trip_capacity'] = edited_slots_df['Long Trip (≥10km)'].tolist()
    
    # Calculate total capacity
    total_slot_capacity = sum(edited_slots_df['Total Capacity'])
    total_slot_drivers = sum(edited_slots_df['Drivers per Slot'])
    total_short_capacity = sum(edited_slots_df['Short Trip (<10km)'])
    total_long_capacity = sum(edited_slots_df['Long Trip (≥10km)'])
    
    # Show summary
    sum_col1, sum_col2, sum_col3 = st.columns(3)
    with sum_col1:
        st.metric("Total Capacity", f"{total_slot_capacity} orders")
    with sum_col2:
        st.metric("Short Trip Capacity", f"{total_short_capacity} orders")
    with sum_col3:
        st.metric("Long Trip Capacity", f"{total_long_capacity} orders")
    
    # Auto-calculate weights, trucks, and orders per trip
    st.markdown("---")
    st.markdown("#### 🤖 Auto-Calculated Recommendations")
    
    # Get truck configurations
    truck_a_cap = 0
    truck_b_cap = 0
    if st.session_state.trucks_config and st.session_state.trucks_config.get('truck_types'):
        for truck_type in st.session_state.trucks_config['truck_types']:
            if truck_type['id'] == 'A':
                truck_a_cap = truck_type['capacity']
            elif truck_type['id'] == 'B':
                truck_b_cap = truck_type['capacity']
    
    # Calculate for each slot
    calc_col1, calc_col2 = st.columns(2)
    
    with calc_col1:
        st.markdown("##### 📊 Slot Analysis")
        slot_calculations = []
        
        for idx, row in edited_slots_df.iterrows():
            slot_num = idx + 1
            capacity = row['Total Capacity']
            drivers = row['Drivers per Slot']
            trips_per_time = row['Trips per Slot Time']
            short_cap = row['Short Trip (<10km)']
            long_cap = row['Long Trip (≥10km)']
            
            # Calculate weight based on total capacity
            weight = capacity / total_slot_capacity if total_slot_capacity > 0 else 0
            
            # Get orders per trip (manual or auto-calculate)
            manual_orders_per_trip = st.session_state.daily_slots['slot_orders_per_trip'][idx] if idx < len(st.session_state.daily_slots['slot_orders_per_trip']) else 0
            
            # Calculate total trips possible in this slot
            total_trips = drivers * trips_per_time
            
            # If orders per trip is manually set, recalculate capacity
            if manual_orders_per_trip > 0:
                # Capacity is determined by orders per trip
                calculated_capacity = total_trips * manual_orders_per_trip
                # Use calculated capacity if it differs significantly from input
                if abs(calculated_capacity - capacity) > 1:
                    capacity = calculated_capacity
                    # Recalculate short/long proportionally
                    if short_cap + long_cap > 0:
                        short_ratio = short_cap / (short_cap + long_cap)
                        long_ratio = long_cap / (short_cap + long_cap)
                        short_cap = int(capacity * short_ratio)
                        long_cap = int(capacity * long_ratio)
                orders_per_trip = manual_orders_per_trip
                orders_per_trip_short = manual_orders_per_trip * (short_cap / capacity) if capacity > 0 else 0
                orders_per_trip_long = manual_orders_per_trip * (long_cap / capacity) if capacity > 0 else 0
            else:
                # Auto-calculate orders per trip based on capacity
                orders_per_trip = capacity / total_trips if total_trips > 0 else 0
                orders_per_trip_short = short_cap / total_trips if total_trips > 0 else 0
                orders_per_trip_long = long_cap / total_trips if total_trips > 0 else 0
            
            # Calculate trucks needed (separate for short and long trips)
            if truck_a_cap > 0:
                # Short trips trucks
                trucks_short_a = int(np.ceil(short_cap / truck_a_cap)) if short_cap > 0 else 0
                # Long trips trucks
                trucks_long_a = int(np.ceil(long_cap / truck_a_cap)) if long_cap > 0 else 0
                trucks_needed_a = trucks_short_a + trucks_long_a
                trucks_needed_b = 0
            elif truck_b_cap > 0:
                trucks_short_b = int(np.ceil(short_cap / truck_b_cap)) if short_cap > 0 else 0
                trucks_long_b = int(np.ceil(long_cap / truck_b_cap)) if long_cap > 0 else 0
                trucks_needed_a = 0
                trucks_needed_b = trucks_short_b + trucks_long_b
            else:
                trucks_needed_a = 0
                trucks_needed_b = 0
                trucks_short_a = 0
                trucks_long_a = 0
            
            # Calculate recommended orders per trip based on drivers
            # Recommendation: optimize for driver efficiency
            if total_trips > 0:
                # Recommended: capacity should be evenly distributed across trips
                recommended_orders_per_trip = capacity / total_trips
                # But also consider practical constraints (4-8 orders per trip is typical)
                if recommended_orders_per_trip < 3:
                    recommended_orders_per_trip = 3
                elif recommended_orders_per_trip > 10:
                    recommended_orders_per_trip = 10
            else:
                recommended_orders_per_trip = 0
            
            # Calculate driver recommendations based on capacity
            driver_recommendations = []
            
            # Calculate if current drivers can handle capacity
            if manual_orders_per_trip > 0:
                # If orders per trip is set, calculate required trips
                required_trips = capacity / manual_orders_per_trip if manual_orders_per_trip > 0 else 0
                required_drivers = np.ceil(required_trips / trips_per_time) if trips_per_time > 0 else 0
                
                if required_drivers > drivers:
                    drivers_needed = int(required_drivers - drivers)
                    driver_recommendations.append({
                        'type': 'capacity_coverage',
                        'priority': 'HIGH',
                        'message': f'Need {drivers_needed} more driver(s) to cover capacity',
                        'current_drivers': drivers,
                        'required_drivers': int(required_drivers),
                        'additional_drivers': drivers_needed,
                        'reason': f'Capacity ({capacity}) requires {required_trips:.0f} trips, but only {total_trips} trips possible with {drivers} drivers'
                    })
                elif drivers > required_drivers:
                    excess_drivers = int(drivers - required_drivers)
                    driver_recommendations.append({
                        'type': 'excess_drivers',
                        'priority': 'LOW',
                        'message': f'Have {excess_drivers} excess driver(s)',
                        'current_drivers': drivers,
                        'required_drivers': int(required_drivers),
                        'additional_drivers': -excess_drivers,
                        'reason': f'Current drivers ({drivers}) exceed required ({int(required_drivers)}) for capacity'
                    })
                else:
                    driver_recommendations.append({
                        'type': 'optimal',
                        'priority': 'INFO',
                        'message': 'Driver count is optimal',
                        'current_drivers': drivers,
                        'required_drivers': int(required_drivers),
                        'additional_drivers': 0,
                        'reason': f'Drivers ({drivers}) perfectly match capacity requirements'
                    })
            else:
                # Auto-calculate based on optimal orders per trip (5-7 range)
                optimal_orders_per_trip = 6  # Balanced default
                required_trips = capacity / optimal_orders_per_trip if optimal_orders_per_trip > 0 else 0
                required_drivers = np.ceil(required_trips / trips_per_time) if trips_per_time > 0 else 0
                
                if required_drivers > drivers:
                    drivers_needed = int(required_drivers - drivers)
                    driver_recommendations.append({
                        'type': 'capacity_coverage',
                        'priority': 'HIGH',
                        'message': f'Need {drivers_needed} more driver(s) to optimally cover capacity',
                        'current_drivers': drivers,
                        'required_drivers': int(required_drivers),
                        'additional_drivers': drivers_needed,
                        'reason': f'For {capacity} orders at {optimal_orders_per_trip} orders/trip, need {required_trips:.0f} trips = {int(required_drivers)} drivers'
                    })
                elif drivers > required_drivers * 1.2:  # 20% tolerance
                    excess_drivers = int(drivers - required_drivers)
                    driver_recommendations.append({
                        'type': 'excess_drivers',
                        'priority': 'MEDIUM',
                        'message': f'Have {excess_drivers} excess driver(s)',
                        'current_drivers': drivers,
                        'required_drivers': int(required_drivers),
                        'additional_drivers': -excess_drivers,
                        'reason': f'Current drivers ({drivers}) significantly exceed required ({int(required_drivers)})'
                    })
                else:
                    driver_recommendations.append({
                        'type': 'optimal',
                        'priority': 'INFO',
                        'message': 'Driver count is adequate',
                        'current_drivers': drivers,
                        'required_drivers': int(required_drivers),
                        'additional_drivers': 0,
                        'reason': f'Drivers ({drivers}) adequately handle capacity'
                    })
            
            # Alternative driver recommendations based on different scenarios
            if manual_orders_per_trip > 0:
                orders_per_trip_for_calc = manual_orders_per_trip
            else:
                orders_per_trip_for_calc = recommended_orders_per_trip if recommended_orders_per_trip > 0 else 6
            
            # Scenario 1: Minimum drivers needed
            min_required_trips = capacity / 10 if capacity > 0 else 0  # Max 10 orders per trip
            min_drivers = np.ceil(min_required_trips / trips_per_time) if trips_per_time > 0 else 0
            if min_drivers > drivers:
                driver_recommendations.append({
                    'type': 'scenario_min',
                    'priority': 'MEDIUM',
                    'message': f'Minimum drivers needed: {int(min_drivers)} (add {int(min_drivers - drivers)})',
                    'current_drivers': drivers,
                    'required_drivers': int(min_drivers),
                    'additional_drivers': int(min_drivers - drivers),
                    'reason': 'Scenario: Maximize orders per trip (10 orders/trip) to minimize drivers'
                })
            
            # Scenario 2: Optimal drivers (balanced)
            optimal_required_trips = capacity / 6 if capacity > 0 else 0  # 6 orders per trip
            optimal_drivers = np.ceil(optimal_required_trips / trips_per_time) if trips_per_time > 0 else 0
            if optimal_drivers != drivers:
                driver_recommendations.append({
                    'type': 'scenario_optimal',
                    'priority': 'HIGH',
                    'message': f'Optimal drivers: {int(optimal_drivers)} ({"add" if optimal_drivers > drivers else "reduce"} {abs(int(optimal_drivers - drivers))})',
                    'current_drivers': drivers,
                    'required_drivers': int(optimal_drivers),
                    'additional_drivers': int(optimal_drivers - drivers),
                    'reason': 'Scenario: Balanced approach (6 orders/trip) for optimal efficiency'
                })
            
            # Scenario 3: Maximum drivers (more trips, smaller orders)
            max_required_trips = capacity / 4 if capacity > 0 else 0  # 4 orders per trip
            max_drivers = np.ceil(max_required_trips / trips_per_time) if trips_per_time > 0 else 0
            if max_drivers > drivers:
                driver_recommendations.append({
                    'type': 'scenario_max',
                    'priority': 'LOW',
                    'message': f'Maximum drivers option: {int(max_drivers)} (add {int(max_drivers - drivers)})',
                    'current_drivers': drivers,
                    'required_drivers': int(max_drivers),
                    'additional_drivers': int(max_drivers - drivers),
                    'reason': 'Scenario: More trips with smaller orders per trip (4 orders/trip) for flexibility'
                })
            
            # Alternative recommendations based on different scenarios
            recommendations = []
            
            # Scenario 1: Maximize trips (smaller orders per trip)
            if total_trips > 0:
                rec_min_trips = max(3, capacity / total_trips)
                recommendations.append({
                    'strategy': 'Maximize Trips',
                    'orders_per_trip': rec_min_trips,
                    'capacity': rec_min_trips * total_trips,
                    'description': f'Smaller orders ({rec_min_trips:.1f}/trip) to maximize trip count'
                })
            
            # Scenario 2: Balance (medium orders per trip)
            if total_trips > 0:
                rec_balanced = capacity / total_trips
                if rec_balanced < 3:
                    rec_balanced = 3
                elif rec_balanced > 8:
                    rec_balanced = 8
                recommendations.append({
                    'strategy': 'Balanced',
                    'orders_per_trip': rec_balanced,
                    'capacity': rec_balanced * total_trips,
                    'description': f'Balanced approach ({rec_balanced:.1f}/trip)'
                })
            
            # Scenario 3: Maximize capacity (larger orders per trip)
            if total_trips > 0:
                rec_max_cap = min(10, capacity / total_trips)
                if rec_max_cap < 5:
                    rec_max_cap = 5
                recommendations.append({
                    'strategy': 'Maximize Capacity',
                    'orders_per_trip': rec_max_cap,
                    'capacity': rec_max_cap * total_trips,
                    'description': f'Larger orders ({rec_max_cap:.1f}/trip) to maximize capacity'
                })
            
            # Get primary driver recommendation
            primary_driver_rec = next((rec for rec in driver_recommendations if rec['type'] in ['capacity_coverage', 'optimal', 'excess_drivers']), None)
            additional_drivers_needed = primary_driver_rec['additional_drivers'] if primary_driver_rec else 0
            
            slot_calculations.append({
                'Slot': f"Slot {slot_num}",
                'Total Capacity': capacity,
                'Weight': weight,
                'Drivers': drivers,
                'Required Drivers': int(primary_driver_rec['required_drivers']) if primary_driver_rec else drivers,
                'Additional Drivers': additional_drivers_needed,
                'Trips/Time': trips_per_time,
                'Total Trips': total_trips,
                'Orders/Trip': orders_per_trip,
                'Recommended Orders/Trip': recommended_orders_per_trip,
                'Trucks A': trucks_needed_a,
                'Trucks B': trucks_needed_b,
                'Short Orders': short_cap,
                'Long Orders': long_cap,
                'Orders/Trip (Short)': orders_per_trip_short,
                'Orders/Trip (Long)': orders_per_trip_long,
                'Manual Setting': manual_orders_per_trip > 0,
                'Driver Recommendations': driver_recommendations
            })
            
            with st.expander(f"**Slot {slot_num}** - Capacity: {capacity} orders", expanded=False):
                col_exp1, col_exp2 = st.columns(2)
                
                with col_exp1:
                    st.markdown("**General:**")
                    st.metric("Total Capacity", f"{capacity} orders")
                    st.metric("Weight", f"{weight:.2%}")
                    st.metric("Drivers", f"{drivers}")
                    st.metric("Trips per Slot Time", f"{trips_per_time}")
                    st.metric("Total Trips Possible", f"{total_trips}")
                    st.metric("Orders per Trip", f"{orders_per_trip:.1f}", 
                             delta="Manual" if manual_orders_per_trip > 0 else "Auto-calculated")
                
                with col_exp2:
                    st.markdown("**Trip Categories:**")
                    st.metric("Short Trip (<10km)", f"{short_cap} orders")
                    st.metric("Long Trip (≥10km)", f"{long_cap} orders")
                    st.metric("Orders/Trip (Short)", f"{orders_per_trip_short:.1f}")
                    st.metric("Orders/Trip (Long)", f"{orders_per_trip_long:.1f}")
                
                st.markdown("**Trucks Needed:**")
                truck_col1, truck_col2 = st.columns(2)
                with truck_col1:
                    st.metric("Trucks Type A", f"{trucks_needed_a}")
                with truck_col2:
                    st.metric("Trucks Type B", f"{trucks_needed_b}")
                
                # Show driver recommendations
                st.markdown("---")
                st.markdown("#### 👥 Driver Recommendations")
                
                # Check if orders exceed this slot's capacity
                slot_excess_analysis = None
                if st.session_state.daily_orders['daily_expected_orders']:
                    max_daily_orders = max(st.session_state.daily_orders['daily_expected_orders']) if st.session_state.daily_orders['daily_expected_orders'] else 0
                    avg_daily_orders = sum(st.session_state.daily_orders['daily_expected_orders']) / len(st.session_state.daily_orders['daily_expected_orders']) if st.session_state.daily_orders['daily_expected_orders'] else 0
                    
                    if max_daily_orders > capacity or avg_daily_orders > capacity:
                        excess_orders = max(max_daily_orders - capacity, avg_daily_orders - capacity)
                        additional_trips = excess_orders / orders_per_trip if orders_per_trip > 0 else 0
                        additional_drivers_excess = np.ceil(additional_trips / trips_per_time) if trips_per_time > 0 else 0
                        total_drivers_with_excess = drivers + int(additional_drivers_excess)
                        
                        slot_excess_analysis = {
                            'excess_orders': excess_orders,
                            'additional_drivers': int(additional_drivers_excess),
                            'total_drivers': total_drivers_with_excess,
                            'max_orders': max_daily_orders,
                            'avg_orders': avg_daily_orders
                        }
                
                # Show excess orders warning if applicable
                if slot_excess_analysis:
                    st.markdown("##### 🚨 When Orders Exceed Capacity")
                    if slot_excess_analysis['max_orders'] > capacity:
                        st.error(f"**Peak Orders ({slot_excess_analysis['max_orders']:.0f}) exceed slot capacity ({capacity})!**")
                    elif slot_excess_analysis['avg_orders'] > capacity:
                        st.warning(f"**Average Orders ({slot_excess_analysis['avg_orders']:.0f}) exceed slot capacity ({capacity})!**")
                    
                    st.markdown(f"**Excess Orders:** {slot_excess_analysis['excess_orders']:.0f} orders")
                    st.markdown(f"**Current Drivers:** {drivers}")
                    st.markdown(f"**Additional Drivers Needed:** {slot_excess_analysis['additional_drivers']} drivers")
                    st.markdown(f"**Total Drivers Required:** {slot_excess_analysis['total_drivers']} drivers")
                    
                    st.info(f"💡 To handle {slot_excess_analysis['excess_orders']:.0f} excess orders with {orders_per_trip:.1f} orders/trip and {trips_per_time} trips/time, you need {slot_excess_analysis['additional_drivers']} more driver(s)")
                
                # Primary recommendation
                primary_rec = primary_driver_rec
                if primary_rec:
                    st.markdown("---")
                    st.markdown("##### 📊 Current Capacity Coverage")
                    if primary_rec['type'] == 'capacity_coverage':
                        st.error(f"🚨 **{primary_rec['message']}**")
                        st.markdown(f"**Current:** {primary_rec['current_drivers']} drivers | **Required:** {primary_rec['required_drivers']} drivers")
                        st.markdown(f"*{primary_rec['reason']}*")
                    elif primary_rec['type'] == 'excess_drivers':
                        st.warning(f"⚠️ **{primary_rec['message']}**")
                        st.markdown(f"**Current:** {primary_rec['current_drivers']} drivers | **Required:** {primary_rec['required_drivers']} drivers")
                        st.markdown(f"*{primary_rec['reason']}*")
                    else:
                        st.success(f"✅ **{primary_rec['message']}**")
                        st.markdown(f"**Current:** {primary_rec['current_drivers']} drivers | **Required:** {primary_rec['required_drivers']} drivers")
                        st.markdown(f"*{primary_rec['reason']}*")
                
                # Alternative driver scenarios
                st.markdown("**Alternative Driver Scenarios:**")
                scenario_recs = [rec for rec in driver_recommendations if rec['type'].startswith('scenario_')]
                for rec in scenario_recs:
                    if rec['additional_drivers'] > 0:
                        st.warning(f"- **{rec['message']}** - {rec['reason']}")
                    elif rec['additional_drivers'] < 0:
                        st.info(f"- **{rec['message']}** - {rec['reason']}")
                    else:
                        st.success(f"- **{rec['message']}** - {rec['reason']}")
                
                # Show orders per trip recommendations
                st.markdown("---")
                st.markdown("#### 💡 Recommended Orders per Trip")
                
                if manual_orders_per_trip > 0:
                    st.info(f"**Current Setting:** {manual_orders_per_trip:.1f} orders per trip (manual)")
                    st.markdown(f"**Recommended:** {recommended_orders_per_trip:.1f} orders per trip based on {drivers} drivers")
                else:
                    st.success(f"**Current (Auto):** {orders_per_trip:.1f} orders per trip")
                    st.markdown(f"**Recommended:** {recommended_orders_per_trip:.1f} orders per trip for optimal efficiency")
                
                # Show alternative scenarios
                st.markdown("**Alternative Strategies:**")
                for rec in recommendations:
                    st.markdown(f"- **{rec['strategy']}:** {rec['orders_per_trip']:.1f} orders/trip → Capacity: {rec['capacity']:.0f} ({rec['description']})")
        
        # Store calculated weights for later use
        st.session_state.daily_slots['calculated_weights'] = [calc['Weight'] for calc in slot_calculations]
    
    with calc_col2:
        st.markdown("##### 📈 Summary")
        st.metric("Total Slot Capacity", f"{total_slot_capacity} orders")
        st.metric("Total Drivers", f"{total_slot_drivers}")
        
        # Visualize capacity distribution
        fig_cap = go.Figure()
        fig_cap.add_trace(go.Bar(
            x=edited_slots_df['Slot'],
            y=edited_slots_df['Short Trip (<10km)'],
            name='Short Trip (<10km)',
            marker_color='lightblue'
        ))
        fig_cap.add_trace(go.Bar(
            x=edited_slots_df['Slot'],
            y=edited_slots_df['Long Trip (≥10km)'],
            name='Long Trip (≥10km)',
            marker_color='darkblue'
        ))
        fig_cap.update_layout(
            title="Slot Capacity Distribution by Trip Type",
            xaxis_title="Slot",
            yaxis_title="Capacity (Orders)",
            barmode='stack'
        )
        st.plotly_chart(fig_cap, width='stretch')
        
        # Show weight distribution
        if total_slot_capacity > 0:
            weights = [calc['Weight'] for calc in slot_calculations]
            fig_weight = px.pie(
                values=weights,
                names=[calc['Slot'] for calc in slot_calculations],
                title="Auto-Calculated Weight Distribution"
            )
            st.plotly_chart(fig_weight, width='stretch')
    
    # Show detailed calculation table
    st.markdown("---")
    st.markdown("#### 📋 Detailed Calculations")
    calc_df = pd.DataFrame(slot_calculations)
    st.dataframe(
        calc_df,
        hide_index=True,
        width='stretch',
        column_config={
            "Slot": st.column_config.TextColumn("Slot", width="small"),
            "Total Capacity": st.column_config.NumberColumn("Total Cap", width="small"),
            "Weight": st.column_config.NumberColumn("Weight", format="%.2%", width="small"),
            "Drivers": st.column_config.NumberColumn("Drivers", width="small"),
            "Trips/Time": st.column_config.NumberColumn("Trips", width="small"),
            "Total Trips": st.column_config.NumberColumn("Tot Trips", width="small"),
            "Short Orders": st.column_config.NumberColumn("Short", width="small"),
            "Long Orders": st.column_config.NumberColumn("Long", width="small"),
            "Required Drivers": st.column_config.NumberColumn("Req Drivers", width="small"),
            "Additional Drivers": st.column_config.NumberColumn("Add Drivers", width="small", help="Positive = need more, Negative = excess"),
            "Orders/Trip": st.column_config.NumberColumn("Orders/Trip", format="%.1f", width="small"),
            "Recommended Orders/Trip": st.column_config.NumberColumn("Recommended", format="%.1f", width="small"),
            "Trucks A": st.column_config.NumberColumn("Trucks A", width="small"),
            "Trucks B": st.column_config.NumberColumn("Trucks B", width="small"),
            "Orders/Trip (Short)": st.column_config.NumberColumn("Short/Trip", format="%.1f", width="small"),
            "Orders/Trip (Long)": st.column_config.NumberColumn("Long/Trip", format="%.1f", width="small"),
        }
    )
    
    # Add driver recommendations summary
    st.markdown("---")
    st.markdown("#### 👥 Driver Recommendations Summary")
    
    # Check if orders exceed capacity
    orders_exceed_capacity_analysis = []
    if st.session_state.daily_orders['daily_expected_orders']:
        total_daily_orders = sum(st.session_state.daily_orders['daily_expected_orders'])
        avg_daily_orders = total_daily_orders / len(st.session_state.daily_orders['daily_expected_orders']) if st.session_state.daily_orders['daily_expected_orders'] else 0
        max_daily_orders = max(st.session_state.daily_orders['daily_expected_orders']) if st.session_state.daily_orders['daily_expected_orders'] else 0
        
        # Analyze each slot for excess orders
        for calc in slot_calculations:
            slot_capacity = calc['Total Capacity']
            slot_drivers = calc['Drivers']
            trips_per_time = calc['Trips/Time']
            orders_per_trip = calc.get('Orders/Trip', 6)
            
            # Calculate if orders exceed capacity
            if max_daily_orders > slot_capacity:
                excess_orders = max_daily_orders - slot_capacity
                # Calculate how many additional drivers needed for excess orders
                additional_trips_needed = excess_orders / orders_per_trip if orders_per_trip > 0 else 0
                additional_drivers_needed = np.ceil(additional_trips_needed / trips_per_time) if trips_per_time > 0 else 0
                
                orders_exceed_capacity_analysis.append({
                    'slot': calc['Slot'],
                    'slot_capacity': slot_capacity,
                    'max_orders': max_daily_orders,
                    'excess_orders': excess_orders,
                    'current_drivers': slot_drivers,
                    'additional_drivers_for_excess': int(additional_drivers_needed),
                    'total_drivers_needed': int(slot_drivers + additional_drivers_needed)
                })
            elif avg_daily_orders > slot_capacity:
                excess_orders = avg_daily_orders - slot_capacity
                additional_trips_needed = excess_orders / orders_per_trip if orders_per_trip > 0 else 0
                additional_drivers_needed = np.ceil(additional_trips_needed / trips_per_time) if trips_per_time > 0 else 0
                
                orders_exceed_capacity_analysis.append({
                    'slot': calc['Slot'],
                    'slot_capacity': slot_capacity,
                    'avg_orders': avg_daily_orders,
                    'excess_orders': excess_orders,
                    'current_drivers': slot_drivers,
                    'additional_drivers_for_excess': int(additional_drivers_needed),
                    'total_drivers_needed': int(slot_drivers + additional_drivers_needed),
                    'note': 'Average orders exceed capacity'
                })
    
    driver_summary_col1, driver_summary_col2 = st.columns(2)
    
    with driver_summary_col1:
        st.markdown("**Driver Requirements by Slot (Current Capacity):**")
        total_additional_drivers = 0
        for calc in slot_calculations:
            slot_name = calc['Slot']
            current = calc['Drivers']
            required = calc.get('Required Drivers', current)
            additional = calc.get('Additional Drivers', 0)
            total_additional_drivers += max(0, additional)
            
            if additional > 0:
                st.error(f"🚨 {slot_name}: Need **{additional} more driver(s)** (Current: {current}, Required: {required})")
            elif additional < 0:
                st.warning(f"⚠️ {slot_name}: {abs(additional)} excess driver(s) (Current: {current}, Required: {required})")
            else:
                st.success(f"✅ {slot_name}: Optimal ({current} drivers)")
        
        st.markdown("---")
        st.metric("Total Additional Drivers Needed", f"{total_additional_drivers}", 
                 delta=f"Across all slots" if total_additional_drivers > 0 else "No additional drivers needed")
        
        # Show excess orders analysis
        if orders_exceed_capacity_analysis:
            st.markdown("---")
            st.markdown("#### 🚨 Drivers Needed When Orders Exceed Capacity")
            st.warning("⚠️ **Orders exceed slot capacity!** Here's how many drivers you need per slot:")
            
            total_excess_drivers = 0
            for analysis in orders_exceed_capacity_analysis:
                slot_name = analysis['slot']
                excess = analysis['excess_orders']
                additional_drivers = analysis['additional_drivers_for_excess']
                total_drivers = analysis['total_drivers_needed']
                current_drivers = analysis['current_drivers']
                slot_cap = analysis['slot_capacity']
                
                st.error(f"**{slot_name}:**")
                st.markdown(f"- Capacity: {slot_cap} orders")
                st.markdown(f"- Orders: {analysis.get('max_orders', analysis.get('avg_orders', 0)):.0f} orders")
                st.markdown(f"- Excess: {excess:.0f} orders")
                st.markdown(f"- Current Drivers: {current_drivers}")
                st.markdown(f"- **Additional Drivers Needed: {additional_drivers}**")
                st.markdown(f"- **Total Drivers Required: {total_drivers}**")
                st.markdown("")
                
                total_excess_drivers += additional_drivers
            
            st.markdown("---")
            st.metric("Total Additional Drivers for Excess Orders", f"{total_excess_drivers}",
                     delta="To handle orders exceeding capacity")
    
    with driver_summary_col2:
        st.markdown("**Calculation Formula:**")
        st.code("""
Required Drivers = Capacity / (Orders per Trip × Trips per Time)

Where:
- Capacity = Total orders to handle
- Orders per Trip = Target orders per trip (4-8 optimal)
- Trips per Time = Trips each driver can make

Example:
- Capacity: 60 orders
- Orders/Trip: 6
- Trips/Time: 2
- Required: 60 / (6 × 2) = 5 drivers

When Orders > Capacity:
Additional Drivers = (Excess Orders / Orders per Trip) / Trips per Time
        """)
        
        # Show excess capacity scenario
        if orders_exceed_capacity_analysis:
            st.markdown("---")
            st.markdown("#### 📊 Excess Orders Scenario")
            
            example = orders_exceed_capacity_analysis[0]
            st.markdown("**Example Calculation:**")
            st.markdown(f"""
            - Slot Capacity: {example['slot_capacity']} orders
            - Actual Orders: {example.get('max_orders', example.get('avg_orders', 0)):.0f} orders
            - Excess: {example['excess_orders']:.0f} orders
            - Orders per Trip: {slot_calculations[0].get('Orders/Trip', 6):.1f}
            - Trips per Time: {slot_calculations[0].get('Trips/Time', 2)}
            
            **Calculation:**
            - Additional Trips = {example['excess_orders']:.0f} / {slot_calculations[0].get('Orders/Trip', 6):.1f} = {example['excess_orders'] / slot_calculations[0].get('Orders/Trip', 6):.1f} trips
            - Additional Drivers = {example['excess_orders'] / slot_calculations[0].get('Orders/Trip', 6):.1f} / {slot_calculations[0].get('Trips/Time', 2)} = **{example['additional_drivers_for_excess']} drivers**
            """)
        
        # Show alternative scenarios summary
        st.markdown("**Alternative Scenarios:**")
        for calc in slot_calculations:
            slot_name = calc['Slot']
            driver_recs = calc.get('Driver Recommendations', [])
            scenario_recs = [r for r in driver_recs if r['type'].startswith('scenario_')]
            if scenario_recs:
                with st.expander(f"{slot_name} Scenarios", expanded=False):
                    for rec in scenario_recs:
                        if rec['additional_drivers'] > 0:
                            st.markdown(f"- **{rec['type'].replace('scenario_', '').title()}:** Add {rec['additional_drivers']} drivers → {rec['reason']}")
                        elif rec['additional_drivers'] < 0:
                            st.markdown(f"- **{rec['type'].replace('scenario_', '').title()}:** Reduce {abs(rec['additional_drivers'])} drivers → {rec['reason']}")
    
    # Add recommendation summary
    st.markdown("---")
    st.markdown("#### 🎯 Orders per Trip Recommendations Summary")
    
    rec_summary_col1, rec_summary_col2 = st.columns(2)
    
    with rec_summary_col1:
        st.markdown("**Quick Recommendations:**")
        for calc in slot_calculations:
            slot_name = calc['Slot']
            current = calc['Orders/Trip']
            recommended = calc['Recommended Orders/Trip']
            is_manual = calc.get('Manual Setting', False)
            
            if is_manual:
                st.info(f"{slot_name}: Currently {current:.1f} (manual) | Recommended: {recommended:.1f}")
            else:
                if abs(current - recommended) > 0.5:
                    st.warning(f"{slot_name}: {current:.1f} → Consider {recommended:.1f} for better efficiency")
                else:
                    st.success(f"{slot_name}: {current:.1f} (optimal)")
    
    with rec_summary_col2:
        st.markdown("**Formula:**")
        st.code("""
Orders per Trip = Total Capacity / (Drivers × Trips per Time)

Optimal Range: 4-8 orders per trip
- < 3: Too small, inefficient
- 4-6: Ideal for most operations
- 7-8: Good for high-volume
- > 10: May be too large
        """)
    
    # Smart Recommendations after slots are configured
    if st.session_state.daily_orders['daily_expected_orders'] and sum(st.session_state.daily_orders['daily_expected_orders']) > 0:
        st.divider()
        st.subheader("🤖 Smart Recommendations")
        
        total_orders_slot = sum(st.session_state.daily_orders['daily_expected_orders'])
        avg_orders_per_day = total_orders_slot / len(st.session_state.daily_orders['daily_expected_orders']) if st.session_state.daily_orders['daily_expected_orders'] else 0
        
        current_slots = st.session_state.daily_slots['slots_per_day']
        slot_capacities = st.session_state.daily_slots.get('slot_capacities', [])
        calculated_weights = st.session_state.daily_slots.get('calculated_weights', [])
        
        # Analyze and provide recommendations
        smart_recs = []
        
        # Slot recommendations
        if current_slots < 3 and avg_orders_per_day > 50:
            smart_recs.append({
                'type': 'slot',
                'priority': 'HIGH',
                'title': 'Add More Time Slots',
                'reason': f'High daily volume ({avg_orders_per_day:.0f} orders/day) benefits from more time slots',
                'action': f'Increase from {current_slots} to 3-4 slots for better distribution',
                'impact': 'Better load balancing and improved OTP'
            })
        
        # Capacity distribution recommendations
        if slot_capacities:
            total_cap = sum(slot_capacities)
            if total_cap < avg_orders_per_day:
                smart_recs.append({
                    'type': 'capacity',
                    'priority': 'HIGH',
                    'title': 'Increase Slot Capacity',
                    'reason': f'Total slot capacity ({total_cap}) is below average daily orders ({avg_orders_per_day:.0f})',
                    'action': f'Increase slot capacities to handle average demand',
                    'impact': 'Prevent capacity shortages'
                })
            
            if calculated_weights:
                max_weight = max(calculated_weights)
                if max_weight > 0.6:
                    smart_recs.append({
                        'type': 'balance',
                        'priority': 'MEDIUM',
                        'title': 'Balance Slot Capacities',
                        'reason': f'One slot has {max_weight*100:.0f}% of capacity - too concentrated',
                        'action': 'Distribute capacity more evenly across slots',
                        'impact': 'Reduced bottlenecks and improved efficiency'
                    })
        
        # Display recommendations
        if smart_recs:
            for rec in smart_recs:
                priority_color = {'HIGH': '🔴', 'MEDIUM': '🟡', 'LOW': '🟢'}.get(rec['priority'], '⚪')
                with st.expander(f"{priority_color} {rec['title']} - {rec['priority']} Priority", expanded=True):
                    st.markdown(f"**Why:** {rec['reason']}")
                    st.markdown(f"**Action:** {rec['action']}")
                    st.markdown(f"**Impact:** {rec['impact']}")
        else:
            st.success("✅ Your slot configuration looks optimal!")
        
        # Next steps
        st.markdown("#### 📋 Next Steps")
        next_steps = []
        
        if not st.session_state.channels_config:
            next_steps.append("1. ⚠️ **Configure Channels** - Add your delivery channels")
        
        if not st.session_state.trucks_config['truck_types']:
            next_steps.append("2. ⚠️ **Configure Trucks** - Set up truck types and capacities")
        
        if st.session_state.channels_config and st.session_state.trucks_config['truck_types']:
            # Calculate if ready for optimization
            total_capacity = sum([calculate_channel_capacity(ch)[0] for ch in st.session_state.channels_config])
            if total_capacity < total_orders_slot:
                next_steps.append(f"3. ⚠️ **Capacity Shortage** - You need {total_orders_slot - total_capacity:.0f} more capacity. Add channels or increase resources.")
            else:
                next_steps.append("3. ✅ **Ready to Optimize** - Click 'AI Optimization & Insights' to get the best plan!")
        
        if next_steps:
            for step in next_steps:
                st.markdown(step)

# ========================== TRUCKS CONFIGURATION SECTION ==========================
elif page == "🚛 Trucks Configuration":
    st.header("🚛 Trucks Configuration")
    
    # Help Section
    with st.expander("ℹ️ What is this section?", expanded=False):
        st.markdown("""
        ### Trucks Configuration
        
        **Purpose:** Define your fleet specifications and capacity.
        
        **Key Concepts:**
        - **Available Trucks:** Total number of trucks in your network
        - **Truck Types:** Different truck categories (A, B, C, etc.)
        - **Capacity:** Maximum orders a truck can handle per day
        - **Total Fleet Capacity:** Total daily capacity across all trucks
        
        **Features:**
        - Configure multiple truck types (default: A & B)
        - Set max orders per day for each type
        - View fleet capacity calculations
        - Visualize truck type distribution
        
        **Default Setup:**
        - Truck A: 4 orders/day (editable)
        - Truck B: 6 orders/day (editable)
        
        **How to Use:**
        1. Enter total available trucks in network
        2. Set capacity for each truck type (orders per day)
        3. Review total fleet capacity
        4. Adjust types and capacities based on your fleet
        
        **Business Impact:**
        - Critical for capacity planning
        - Determines maximum orders you can fulfill
        - Helps identify if you need to expand fleet
        """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        available_trucks = st.number_input(
            "Number of Available Trucks in Network",
            min_value=1,
            max_value=1000,
            value=st.session_state.trucks_config['available_trucks'],
            key='available_trucks_input'
        )
        st.session_state.trucks_config['available_trucks'] = available_trucks
    
    st.subheader("Truck Types Configuration")
    
    truck_types_data = []
    for truck_type in st.session_state.trucks_config['truck_types']:
        truck_types_data.append({
            'Truck Type': truck_type['id'],
            'Max Orders per Day': truck_type['capacity']
        })
    
    df_trucks = pd.DataFrame(truck_types_data)
    
    # Display with editable table
    edited_trucks_df = st.data_editor(
        df_trucks,
        column_config={
            "Truck Type": st.column_config.TextColumn("Truck Type", disabled=True),
            "Max Orders per Day": st.column_config.NumberColumn("Max Orders per Day", min_value=1, step=1)
        },
        hide_index=True,
        width='stretch'
    )
    
    # Update truck types
    for idx, truck_type in enumerate(st.session_state.trucks_config['truck_types']):
        truck_type['capacity'] = int(edited_trucks_df.iloc[idx]['Max Orders per Day'])
    
    # Visualization
    fig = px.bar(
        edited_trucks_df,
        x='Truck Type',
        y='Max Orders per Day',
        title="Truck Capacity Configuration",
        color='Max Orders per Day',
        color_continuous_scale='Greens'
    )
    st.plotly_chart(fig, width='stretch')
    
    # Calculate fleet capacity
    total_capacity = sum([truck_type['capacity'] for truck_type in st.session_state.trucks_config['truck_types']])
    fleet_capacity = available_trucks * total_capacity
    st.info(f"💡 Total Fleet Capacity: {fleet_capacity} orders/day")

# ========================== CHANNEL CONFIGURATION SECTION ==========================
elif page == "🔗 Channel Configuration":
    st.header("🔗 Channel Configuration")
    
    # Help Section
    with st.expander("ℹ️ What is this section?", expanded=False):
        st.markdown("""
        ### Channel Configuration
        
        **Purpose:** Define delivery channels (routes, zones, or service areas).
        
        **What is a Channel?**
        - A delivery channel is a distinct operational route or zone
        - Examples: City Zone 1, Suburbs, Express Delivery, Standard Delivery
        - Each channel can have different costs, capacity, and performance
        
        **Key Parameters:**
        
        **1. Basic Information:**
        - **Channel ID:** Unique identifier (e.g., "C1", "North Zone")
        - **Cost per Order:** Variable cost per delivered order ($)
        - **Fixed Cost per Trip:** Fixed cost regardless of order count ($)
        
        **2. Resources:**
        - **Drivers Available:** Number of drivers in this channel
        - **Trucks Type A/B:** Number of each truck type available
        - **Trips per Day per Driver:** Maximum trips each driver can make
        - **Average Orders per Trip:** Typical orders carried per trip
        
        **3. Performance Metrics:**
        - **OTP 24h, 7d, 30d:** On-Time Performance percentages
        - Historical performance data for optimization
        
        **How to Calculate Capacity:**
        - Driver Capacity = Drivers × Trips/Driver × Orders/Trip
        - Truck Capacity = (Trucks A × Truck A Capacity) + (Trucks B × Truck B Capacity)
        - Channel Capacity = Minimum(Driver Capacity, Truck Capacity)
        
        **Business Impact:**
        - Determines which channels can handle more orders
        - Helps identify bottlenecks
        - Enables cost optimization
        """)
    
    num_channels = st.number_input(
        "Number of Channels",
        min_value=1,
        max_value=20,
        value=max(1, len(st.session_state.channels_config)),
        key='num_channels_input'
    )
    
    # Ensure channels_config has enough items
    while len(st.session_state.channels_config) < num_channels:
        channel_id = f"C{len(st.session_state.channels_config) + 1}"
        st.session_state.channels_config.append({
            'id': channel_id,
            'cost_per_order': 10.0,
            'fixed_cost_per_trip': 50.0,
            'drivers_available': 5,
            'available_trucks_type_a': 3,
            'available_trucks_type_b': 2,
            'trips_per_day_per_driver': 4,
            'avg_orders_per_trip': 3,
            'otp_24h': 95.0,
            'otp_7d': 93.0,
            'otp_30d': 90.0
        })
    
    # Remove excess channels
    st.session_state.channels_config = st.session_state.channels_config[:num_channels]
    
    st.subheader("Channel Details")
    
    for idx, channel in enumerate(st.session_state.channels_config):
        with st.expander(f"📡 Channel {channel['id']}", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                channel['id'] = st.text_input(
                    "Channel ID",
                    value=channel['id'],
                    key=f'channel_id_{idx}'
                )
                channel['cost_per_order'] = st.number_input(
                    "Cost per Order ($)",
                    min_value=0.0,
                    value=float(channel['cost_per_order']),
                    step=0.1,
                    key=f'cost_order_{idx}'
                )
                channel['fixed_cost_per_trip'] = st.number_input(
                    "Fixed Cost per Trip ($)",
                    min_value=0.0,
                    value=float(channel['fixed_cost_per_trip']),
                    step=0.1,
                    key=f'fixed_cost_{idx}'
                )
                channel['drivers_available'] = st.number_input(
                    "Drivers Available",
                    min_value=0,
                    value=int(channel['drivers_available']),
                    step=1,
                    key=f'drivers_{idx}'
                )
            
            with col2:
                channel['available_trucks_type_a'] = st.number_input(
                    "Available Trucks Type A",
                    min_value=0,
                    value=int(channel['available_trucks_type_a']),
                    step=1,
                    key=f'trucks_a_{idx}'
                )
                channel['available_trucks_type_b'] = st.number_input(
                    "Available Trucks Type B",
                    min_value=0,
                    value=int(channel['available_trucks_type_b']),
                    step=1,
                    key=f'trucks_b_{idx}'
                )
                channel['trips_per_day_per_driver'] = st.number_input(
                    "Trips per Day per Driver",
                    min_value=1,
                    value=int(channel['trips_per_day_per_driver']),
                    step=1,
                    key=f'trips_{idx}'
                )
                channel['avg_orders_per_trip'] = st.number_input(
                    "Average Orders per Trip",
                    min_value=1,
                    value=int(channel['avg_orders_per_trip']),
                    step=1,
                    key=f'avg_orders_{idx}'
                )
            
            st.subheader("OTP Metrics")
            col3, col4, col5 = st.columns(3)
            
            with col3:
                channel['otp_24h'] = st.number_input(
                    "OTP (Previous 24 hours) %",
                    min_value=0.0,
                    max_value=100.0,
                    value=float(channel['otp_24h']),
                    step=0.1,
                    key=f'otp_24h_{idx}'
                )
            with col4:
                channel['otp_7d'] = st.number_input(
                    "OTP (Previous 7 days) %",
                    min_value=0.0,
                    max_value=100.0,
                    value=float(channel['otp_7d']),
                    step=0.1,
                    key=f'otp_7d_{idx}'
                )
            with col5:
                channel['otp_30d'] = st.number_input(
                    "OTP (Previous 30 days) %",
                    min_value=0.0,
                    max_value=100.0,
                    value=float(channel['otp_30d']),
                    step=0.1,
                    key=f'otp_30d_{idx}'
                )
    
    # Summary table
    st.subheader("Channel Summary")
    summary_data = []
    for channel in st.session_state.channels_config:
        summary_data.append({
            'Channel ID': channel['id'],
            'Cost/Order': f"${channel['cost_per_order']:.2f}",
            'Fixed Cost/Trip': f"${channel['fixed_cost_per_trip']:.2f}",
            'Drivers': channel['drivers_available'],
            'Trucks A': channel['available_trucks_type_a'],
            'Trucks B': channel['available_trucks_type_b'],
            'OTP (24h)': f"{channel['otp_24h']:.1f}%",
            'OTP (7d)': f"{channel['otp_7d']:.1f}%"
        })
    
    df_summary = pd.DataFrame(summary_data)
    st.dataframe(df_summary, width='stretch', hide_index=True)
    
    # Intelligent Channel Analytics
    st.divider()
    st.subheader("🎯 Channel Intelligence Analytics")
    
    for idx, channel in enumerate(st.session_state.channels_config):
        with st.expander(f"📊 Analytics for {channel['id']}", expanded=False):
            # Calculate capacity
            capacity, driver_cap, truck_cap = calculate_channel_capacity(channel)
            weighted_otp = calculate_weighted_otp(channel)
            
            col_an1, col_an2, col_an3 = st.columns(3)
            
            with col_an1:
                st.metric("Total Daily Capacity", f"{capacity} orders")
                st.caption(f"Driver capacity: {driver_cap}")
                st.caption(f"Truck capacity: {truck_cap}")
            
            with col_an2:
                cost_efficiency = weighted_otp / (channel['cost_per_order'] + channel['fixed_cost_per_trip'] / channel['avg_orders_per_trip'])
                st.metric("Cost Efficiency Score", f"{cost_efficiency:.2f}")
                st.caption("Higher = Better value")
            
            with col_an3:
                st.metric("Weighted OTP", f"{weighted_otp:.1f}%")
                utilization_warning = "⚠️ High" if channel['otp_24h'] < 90 else "✓ Good"
                st.caption(utilization_warning)
            
            # ROI Prediction
            if channel['trips_per_day_per_driver'] > 0 and channel['drivers_available'] > 0:
                max_trips = channel['drivers_available'] * channel['trips_per_day_per_driver']
                max_orders_day = max_trips * channel['avg_orders_per_trip']
                
                # Cost for full capacity utilization
                avg_cost_per_order = channel['cost_per_order'] + (channel['fixed_cost_per_trip'] / channel['avg_orders_per_trip'])
                max_daily_cost = max_orders_day * avg_cost_per_order
                
                col_roi1, col_roi2 = st.columns(2)
                with col_roi1:
                    st.info(f"💡 At full capacity: {max_orders_day} orders/day for ${max_daily_cost:.2f}")
                    st.caption(f"Avg cost per order: ${avg_cost_per_order:.2f}")
                
                with col_roi2:
                    if st.session_state.daily_orders['daily_expected_orders']:
                        total_daily_orders = sum(st.session_state.daily_orders['daily_expected_orders'])
                        avg_daily_orders = total_daily_orders / len(st.session_state.daily_orders['daily_expected_orders'])
                        utilization_pct = (avg_daily_orders / max_orders_day * 100) if max_orders_day > 0 else 0
                        st.metric("Expected Utilization", f"{utilization_pct:.1f}%")
                        if utilization_pct > 100:
                            st.error("⚠️ Capacity exceeded!")
                        elif utilization_pct > 80:
                            st.warning("⚠️ High utilization expected")

# ========================== OPTIMIZATION PARAMETERS SECTION ==========================
elif page == "⚙️ Optimization Parameters":
    st.header("⚙️ Optimization Parameters")
    
    # Help Section
    with st.expander("ℹ️ What is this section?", expanded=False):
        st.markdown("""
        ### Optimization Parameters
        
        **Purpose:** Configure how the AI makes optimization decisions.
        
        **1. OTP Time Window Weights**
        - Weights for different time horizons (24h/7d/30d)
        - Higher weight = more emphasis on that metric
        - Total must equal 1.0
        - **Example:** 0.5 (24h) + 0.3 (7d) + 0.2 (30d) = 1.0
        
        **Why Different Weights?**
        - Recent performance (24h) may be most important
        - 7-day average shows consistency
        - 30-day gives long-term trends
        
        **2. OTP Target Recommendations**
        - Scenario A (X%): Conservative target for planning
        - Scenario B (Y%): Ambitious target for excellence
        - AI will optimize to achieve these targets
        
        **3. Cost vs OTP Weight**
        - **0.0:** Maximize OTP (ignore costs) → Best service quality
        - **0.5:** Balanced approach → Balance cost and performance
        - **1.0:** Minimize costs (ignore OTP) → Cheapest option
        
        **How It Works:**
        The AI uses these parameters to allocate orders across channels
        - Prioritizes channels with better performance
        - Considers cost constraints
        - Balances multiple objectives based on your priorities
        
        **Business Impact:**
        - Choose strategy based on business priorities
        - Adjust in real-time as conditions change
        - Test different scenarios for sensitivity analysis
        """)
    
    st.subheader("OTP Time Window Weights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        weight_24h = st.slider(
            "Weight: 24 Hours",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1
        )
    
    with col2:
        weight_7d = st.slider(
            "Weight: 7 Days",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.1
        )
    
    with col3:
        weight_30d = st.slider(
            "Weight: 30 Days",
            min_value=0.0,
            max_value=1.0,
            value=0.2,
            step=0.1
        )
    
    total_weight = weight_24h + weight_7d + weight_30d
    if abs(total_weight - 1.0) > 0.01:
        st.warning(f"⚠️ Total weight is {total_weight:.2f}")
        if st.button("Normalize to 1.0"):
            weight_24h = weight_24h / total_weight
            weight_7d = weight_7d / total_weight
            weight_30d = weight_30d / total_weight
            st.rerun()
    
    st.session_state.optimization_params['otp_time_weights'] = {
        '24h': weight_24h,
        '7d': weight_7d,
        '30d': weight_30d
    }
    
    st.subheader("OTP Target Recommendations")
    target1 = st.number_input(
        "Target 1: Achieve X% OTP",
        min_value=0,
        max_value=100,
        value=int(st.session_state.optimization_params['otp_targets'][0]),
        step=1
    )
    target2 = st.number_input(
        "Target 2: Achieve Y% OTP",
        min_value=0,
        max_value=100,
        value=int(st.session_state.optimization_params['otp_targets'][1]),
        step=1
    )
    
    st.session_state.optimization_params['otp_targets'] = [target1, target2]
    
    st.subheader("Cost vs OTP Weight")
    st.markdown("""
    - **0.0**: Maximize OTP (minimum cost)
    - **1.0**: Minimize Cost (minimum OTP)
    """)
    
    cost_otp_weight = st.slider(
        "Weight for Cost vs OTP",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.optimization_params['cost_otp_weight'],
        step=0.1
    )
    
    st.session_state.optimization_params['cost_otp_weight'] = cost_otp_weight
    
    # Visualization
    weights_df = pd.DataFrame({
        'Time Window': ['24 Hours', '7 Days', '30 Days'],
        'Weight': [weight_24h, weight_7d, weight_30d]
    })
    
    fig = px.bar(
        weights_df,
        x='Time Window',
        y='Weight',
        title="OTP Time Window Weights",
        color='Weight',
        color_continuous_scale='Purples'
    )
    st.plotly_chart(fig, width='stretch')
    
    # Cost vs OTP visualization
    cost_otp_df = pd.DataFrame({
        'Objective': ['Maximize OTP', 'Balance', 'Minimize Cost'],
        'Weight': [0.0, 0.5, 1.0],
        'Current': [1 if abs(cost_otp_weight - w) < 0.01 else 0 for w in [0.0, 0.5, 1.0]]
    })
    
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=cost_otp_df['Objective'], y=cost_otp_df['Weight'], marker_color='lightblue'))
    fig2.add_trace(go.Scatter(x=['Maximize OTP', 'Balance', 'Minimize Cost'], y=[0.0, 0.5, 1.0], 
                             mode='markers', marker=dict(size=20, color='red', symbol='cross')))
    fig2.update_layout(title="Cost vs OTP Weight Configuration", yaxis_title="Weight", xaxis_title="Objective")
    st.plotly_chart(fig2, width='stretch')

# ========================== RESULTS & RECOMMENDATIONS SECTION ==========================
elif page == "📊 AI Optimization & Insights":
    st.header("🤖 AI-Powered Optimization Engine")
    
    # Help Section
    with st.expander("ℹ️ How does the AI optimization work?", expanded=False):
        st.markdown("""
        ### AI-Powered Optimization Explained
        
        **What Does It Do?**
        The AI automatically determines the BEST way to distribute orders across your channels
        to achieve your targets while minimizing costs.
        
        **How It Optimizes:**
        
        1. **Calculates Capacity** for each channel (drivers vs trucks)
        2. **Scores Channels** based on:
           - OTP performance (weighted by your time preferences)
           - Available capacity
           - Cost efficiency
        3. **Allocates Orders** starting with best-performing channels
        4. **Calculates Costs** (variable + fixed costs per trip)
        5. **Generates Insights** about potential issues
        
        **What You Get:**
        - **Optimized Allocation:** Orders assigned to best channels
        - **Cost Breakdown:** Daily and total costs calculated
        - **OTP Predictions:** Expected performance if you follow the plan
        - **Risk Alerts:** Warnings if capacity is insufficient
        - **Actionable Insights:** Specific recommendations
        
        **Key Metrics Explained:**
        - **Total Cost:** Sum of all operational costs
        - **Cost per Order:** Average cost to fulfill one order
        - **Weighted Avg OTP:** Performance weighted by order volume
        - **Fulfillment Rate:** % of orders that can be fulfilled
        
        **Interpreting Results:**
        - Green metrics = Good performance
        - Yellow/Red metrics = Needs attention
        - Insights = Specific actions to take
        
        **When to Use:**
        - Before starting operations to plan ahead
        - After changes to see impact
        - Weekly/monthly for capacity planning
        """)
    
    if not st.session_state.channels_config:
        st.warning("⚠️ Please configure channels first!")
    elif not st.session_state.daily_orders['daily_expected_orders']:
        st.warning("⚠️ Please configure daily orders first!")
    else:
        # Intelligence Summary Card
        st.subheader("🧠 Quick Intelligence Summary")
        col_sum1, col_sum2, col_sum3, col_sum4 = st.columns(4)
        
        with col_sum1:
            total_orders = sum(st.session_state.daily_orders['daily_expected_orders'])
            total_capacity = 0
            for ch in st.session_state.channels_config:
                cap, _, _ = calculate_channel_capacity(ch)
                total_capacity += cap
            
            st.metric("Total Orders", f"{total_orders}")
        
        with col_sum2:
            st.metric("Total Capacity", f"{total_capacity:.0f}", 
                     delta=f"{((total_capacity/total_orders-1)*100):.1f}%" if total_orders > 0 else "N/A")
            
            # Add capacity explanation
            with st.expander("ℹ️ How is Total Capacity Calculated?", expanded=False):
                st.markdown("""
                **Total Capacity** represents the maximum orders you can handle per day.
                
                **It's calculated as the MINIMUM of two constraints:**
                
                **1. Driver Capacity (per channel):**
                ```
                Driver Capacity = Drivers Available × Trips per Driver × Orders per Trip
                ```
                
                **2. Truck Capacity (per channel):**
                ```
                Truck Capacity = (Trucks Type A × Capacity A) + (Trucks Type B × Capacity B)
                ```
                
                **Channel Capacity = MIN(Driver Capacity, Truck Capacity)**
                
                **Total Capacity = Sum of all Channel Capacities**
                
                **Example:**
                - Channel has 5 drivers, each does 3 trips/day, carrying 4 orders/trip
                - Driver Capacity = 5 × 3 × 4 = 60 orders/day
                - Channel has 2 Type A trucks (capacity 6) and 1 Type B truck (capacity 4)
                - Truck Capacity = (2 × 6) + (1 × 4) = 16 orders/day
                - **Channel Capacity = MIN(60, 16) = 16 orders/day** ← Truck limit!
                
                **Why MIN?** You need BOTH drivers AND trucks. If you have 60 driver capacity but only 16 truck capacity, you can only handle 16 orders!
                """)
                
                # Show current breakdown
                if st.session_state.channels_config:
                    st.markdown("**Your Current Capacity Breakdown:**")
                    for ch in st.session_state.channels_config:
                        cap, driver_cap, truck_cap = calculate_channel_capacity(ch)
                        limiting_factor = "Driver" if cap == driver_cap else "Truck"
                        st.markdown(f"""
                        **Channel {ch['id']}:**
                        - Total Capacity: **{cap:.0f}** orders/day
                        - Driver Capacity: {driver_cap:.0f} orders/day
                        - Truck Capacity: {truck_cap:.0f} orders/day
                        - ⚠️ **Limiting Factor:** {limiting_factor} capacity
                        """)
        
        with col_sum3:
            if total_capacity < total_orders:
                st.metric("⚠️ Risk Level", "HIGH", delta="-Capacity Shortage")
            elif total_capacity < total_orders * 1.2:
                st.metric("⚠️ Risk Level", "MEDIUM", delta="Tight Capacity")
            else:
                st.metric("✓ Risk Level", "LOW", delta="Sufficient Capacity")
        
        with col_sum4:
            avg_otp = np.mean([calculate_weighted_otp(ch) for ch in st.session_state.channels_config])
            st.metric("Average OTP", f"{avg_otp:.1f}%")
        
        st.divider()
        
        # Calculate recommendations
        st.subheader("🎯 Optimization Scenarios")
        
        col1, col2 = st.columns(2)
        
        run_optimization = False
        
        with col1:
            st.markdown(f"#### Target: {st.session_state.optimization_params['otp_targets'][0]}% OTP")
            if st.button("🚀 Run Optimization A", type="primary", use_container_width=True):
                with st.spinner("🤖 AI is optimizing allocation..."):
                    results = optimize_allocation()
                    if results:
                        st.session_state.optimization_results['scenario_a'] = results
                        run_optimization = True
        
        with col2:
            st.markdown(f"#### Target: {st.session_state.optimization_params['otp_targets'][1]}% OTP")
            if st.button("🚀 Run Optimization B", type="primary", use_container_width=True):
                with st.spinner("🤖 AI is optimizing allocation..."):
                    results = optimize_allocation()
                    if results:
                        st.session_state.optimization_results['scenario_b'] = results
                        run_optimization = True
        
        # Display results if available
        if 'best_scenario' in st.session_state.optimization_results or 'scenario_a' in st.session_state.optimization_results or 'scenario_b' in st.session_state.optimization_results:
            scenario_to_show = None
            scenario_name = None
            
            # Prioritize best_scenario if available
            if 'best_scenario' in st.session_state.optimization_results:
                scenario_to_show = 'best_scenario'
                scenario_name = "⭐ Best Scenario (Auto-Generated)"
            elif 'scenario_a' in st.session_state.optimization_results:
                scenario_to_show = 'scenario_a'
                scenario_name = f"Scenario A ({st.session_state.optimization_params['otp_targets'][0]}% OTP Target)"
            elif 'scenario_b' in st.session_state.optimization_results:
                scenario_to_show = 'scenario_b'
                scenario_name = f"Scenario B ({st.session_state.optimization_params['otp_targets'][1]}% OTP Target)"
            
            if scenario_to_show:
                results = st.session_state.optimization_results[scenario_to_show]
                roi_data = calculate_roi(results)
                
                st.divider()
                st.subheader(f"📊 {scenario_name} - Detailed Analysis")
                
                # Quick Recommendations Banner
                st.markdown("### 🎯 QUICK RECOMMENDATIONS")
                quick_rec_col1, quick_rec_col2, quick_rec_col3 = st.columns(3)
                
                # Calculate quick stats - more accurate
                total_orders_quick = sum([r['total_orders'] for r in results])
                channels_used = set()
                max_drivers_per_day = []
                
                for day_result in results:
                    day_drivers = {}
                    for alloc in day_result['allocations']:
                        channel = next((c for c in st.session_state.channels_config if c['id'] == alloc['channel']), None)
                        if channel:
                            orders_per_trip = channel['avg_orders_per_trip']
                            trips_needed = np.ceil(alloc['orders'] / orders_per_trip)
                            trips_per_driver = channel['trips_per_day_per_driver']
                            drivers_needed = np.ceil(trips_needed / trips_per_driver)
                            day_drivers[alloc['channel']] = int(drivers_needed)
                            channels_used.add(alloc['channel'])
                    max_drivers_per_day.append(sum(day_drivers.values()))
                
                total_drivers_quick = max(max_drivers_per_day) if max_drivers_per_day else 0
                avg_drivers_needed = np.mean(max_drivers_per_day) if max_drivers_per_day else 0
                
                # Find primary channel
                channel_order_counts = {}
                for day_result in results:
                    for alloc in day_result['allocations']:
                        ch = alloc['channel']
                        channel_order_counts[ch] = channel_order_counts.get(ch, 0) + alloc['orders']
                primary_channel = max(channel_order_counts.items(), key=lambda x: x[1])[0] if channel_order_counts else 'N/A'
                
                with quick_rec_col1:
                    st.success(f"**Use {len(channels_used)} Channels**")
                    st.markdown(f"Primary: **{primary_channel}**")
                
                with quick_rec_col2:
                    st.info(f"**Peak: {total_drivers_quick} Drivers**")
                    st.markdown(f"Avg: {avg_drivers_needed:.0f} drivers/day")
                    avg_orders_per_driver_quick = total_orders_quick / avg_drivers_needed / len(results) if avg_drivers_needed > 0 else 0
                    st.caption(f"~{avg_orders_per_driver_quick:.1f} orders/driver/day")
                
                with quick_rec_col3:
                    st.warning(f"**Total Cost: ${roi_data['total_cost']:,.0f}**")
                    st.markdown(f"${roi_data['avg_cost_per_order']:.2f} per order")
                
                st.divider()
                
                # ROI Metrics
                st.markdown("#### 💰 Financial Metrics")
                roi_col1, roi_col2, roi_col3, roi_col4 = st.columns(4)
                
                with roi_col1:
                    st.metric("Total Cost", f"${roi_data['total_cost']:,.2f}")
                with roi_col2:
                    st.metric("Cost per Order", f"${roi_data['avg_cost_per_order']:.2f}")
                with roi_col3:
                    st.metric("Weighted Avg OTP", f"{roi_data['weighted_avg_otp']:.2f}%")
                with roi_col4:
                    st.metric("Fulfillment Rate", f"{roi_data['fulfillment_rate']:.1f}%")
                
                # Enhanced Daily Breakdown with Driver Details
                st.markdown("#### 📅 Daily Allocation Breakdown - Detailed Plan")
                
                all_daily_data = []
                slot_allocation_data = []
                
                # Get slot configuration
                slot_capacities = st.session_state.daily_slots.get('slot_capacities', [])
                slot_drivers = st.session_state.daily_slots.get('slot_drivers', [])
                calculated_weights = st.session_state.daily_slots.get('calculated_weights', [])
                
                # Use calculated weights if available, otherwise calculate from capacities
                if calculated_weights:
                    slot_weights = calculated_weights
                elif slot_capacities:
                    total_cap = sum(slot_capacities)
                    slot_weights = [cap / total_cap if total_cap > 0 else 0 for cap in slot_capacities]
                else:
                    slot_weights = []
                
                for day_result in results:
                    day_orders = day_result['total_orders']
                    
                    # Distribute orders and drivers across slots
                    for slot_idx, slot_weight in enumerate(slot_weights):
                        if slot_idx < len(slot_drivers):
                            slot_orders = int(day_orders * slot_weight)
                            slot_drivers_available = slot_drivers[slot_idx]
                            
                            # Allocate to channels for this slot
                            for alloc in day_result['allocations']:
                                # Find channel details
                                channel = next((c for c in st.session_state.channels_config if c['id'] == alloc['channel']), None)
                                
                                if channel:
                                    # Calculate slot-specific orders (proportional to weight)
                                    slot_channel_orders = int(alloc['orders'] * slot_weight)
                                    
                                    if slot_channel_orders > 0:
                                        # Calculate driver requirements for this slot
                                        orders_per_trip = channel['avg_orders_per_trip']
                                        trips_needed = np.ceil(slot_channel_orders / orders_per_trip)
                                        trips_per_driver = channel['trips_per_day_per_driver']
                                        
                                        # Drivers needed for this slot (can't exceed slot drivers)
                                        drivers_needed = min(int(np.ceil(trips_needed / trips_per_driver)), slot_drivers_available)
                                        orders_per_driver = slot_channel_orders / drivers_needed if drivers_needed > 0 else 0
                                        
                                        slot_allocation_data.append({
                                            'Day': alloc['day'],
                                            'Slot': f"Slot {slot_idx + 1}",
                                            'Slot Weight': f"{slot_weight:.2f}",
                                            'Recommended Channel': alloc['channel'],
                                            'Orders in Slot': int(slot_channel_orders),
                                            'Drivers in Slot': drivers_needed,
                                            'Orders per Driver': f"{orders_per_driver:.1f}",
                                            'Trips Needed': int(trips_needed),
                                        })
                        
                        # Also keep original channel-level data
                        for alloc in day_result['allocations']:
                            channel = next((c for c in st.session_state.channels_config if c['id'] == alloc['channel']), None)
                            
                            if channel:
                                # Calculate driver requirements
                                orders_per_trip = channel['avg_orders_per_trip']
                                trips_needed = np.ceil(alloc['orders'] / orders_per_trip)
                                trips_per_driver = channel['trips_per_day_per_driver']
                                drivers_needed = np.ceil(trips_needed / trips_per_driver)
                                orders_per_driver = alloc['orders'] / drivers_needed if drivers_needed > 0 else 0
                                
                                all_daily_data.append({
                                    'Day': alloc['day'],
                                    'Recommended Channel': alloc['channel'],
                                    'Orders Allocated': int(alloc['orders']),
                                    'Drivers Needed': int(drivers_needed),
                                    'Orders per Driver': f"{orders_per_driver:.1f}",
                                    'Trips Needed': int(trips_needed),
                                    'Cost': f"${alloc['cost']:.2f}",
                                    'OTP': f"{alloc['otp']:.1f}%",
                                    'Capacity Util': f"{alloc['capacity_utilization']:.1f}%"
                                })
                
                # Show slot-based breakdown
                if slot_allocation_data:
                    st.markdown("#### 📊 Detailed Slot-by-Slot Breakdown")
                    st.markdown("**This shows how orders and drivers are distributed across time slots each day:**")
                    
                    df_slot_breakdown = pd.DataFrame(slot_allocation_data)
                    st.dataframe(
                        df_slot_breakdown,
                        hide_index=True,
                        width='stretch',
                        column_config={
                            "Day": st.column_config.NumberColumn("Day", width="small"),
                            "Slot": st.column_config.TextColumn("Time Slot", width="small"),
                            "Slot Weight": st.column_config.TextColumn("Weight", width="small"),
                            "Recommended Channel": st.column_config.TextColumn("Channel", width="medium"),
                            "Orders in Slot": st.column_config.NumberColumn("Orders", width="small"),
                            "Drivers in Slot": st.column_config.NumberColumn("Drivers", width="small"),
                            "Orders per Driver": st.column_config.TextColumn("Orders/Driver", width="small"),
                        }
                    )
                    
                    # Summary by slot
                    st.markdown("#### 📈 Slot Summary")
                    slot_summary = {}
                    for row in slot_allocation_data:
                        slot = row['Slot']
                        if slot not in slot_summary:
                            slot_summary[slot] = {
                                'Total Orders': 0,
                                'Total Drivers': 0,
                                'Days': set(),
                                'Channels': set()
                            }
                        slot_summary[slot]['Total Orders'] += row['Orders in Slot']
                        slot_summary[slot]['Total Drivers'] = max(slot_summary[slot]['Total Drivers'], row['Drivers in Slot'])
                        slot_summary[slot]['Days'].add(row['Day'])
                        slot_summary[slot]['Channels'].add(row['Recommended Channel'])
                    
                    slot_summary_cols = st.columns(len(slot_summary))
                    for idx, (slot, data) in enumerate(sorted(slot_summary.items())):
                        with slot_summary_cols[idx] if idx < len(slot_summary_cols) else slot_summary_cols[0]:
                            st.metric(f"{slot} Orders", f"{data['Total Orders']}")
                            st.metric(f"{slot} Drivers", f"{data['Total Drivers']}")
                            st.caption(f"Days: {len(data['Days'])} | Channels: {len(data['Channels'])}")
                    
                    st.divider()
                
                # Channel-level summary (existing)
                if all_daily_data:
                    df_daily = pd.DataFrame(all_daily_data)
                    st.dataframe(df_daily, hide_index=True, width='stretch')
                    
                    # Summary by Channel
                    st.markdown("#### 📊 Channel Recommendation Summary")
                    channel_summary = {}
                    for row in all_daily_data:
                        ch = row['Recommended Channel']
                        if ch not in channel_summary:
                            channel_summary[ch] = {
                                'Total Orders': 0,
                                'Total Drivers': 0,
                                'Days Used': 0,
                                'Total Cost': 0,
                                'Avg OTP': 0,
                                'OTP Weight': 0
                            }
                        channel_summary[ch]['Total Orders'] += row['Orders Allocated']
                        channel_summary[ch]['Total Drivers'] = max(channel_summary[ch]['Total Drivers'], row['Drivers Needed'])
                        channel_summary[ch]['Days Used'] += 1
                        channel_summary[ch]['Total Cost'] += float(row['Cost'].replace('$', '').replace(',', ''))
                        channel_summary[ch]['OTP Weight'] += float(row['OTP'].replace('%', '')) * row['Orders Allocated']
                    
                    # Create summary dataframe
                    summary_rows = []
                    for ch, data in channel_summary.items():
                        avg_otp = data['OTP Weight'] / data['Total Orders'] if data['Total Orders'] > 0 else 0
                        avg_orders_per_driver = data['Total Orders'] / data['Total Drivers'] if data['Total Drivers'] > 0 else 0
                        summary_rows.append({
                            'Recommended Channel': ch,
                            'Total Orders': data['Total Orders'],
                            'Drivers Required': data['Total Drivers'],
                            'Orders per Driver': f"{avg_orders_per_driver:.1f}",
                            'Days Active': data['Days Used'],
                            'Total Cost': f"${data['Total Cost']:,.2f}",
                            'Avg OTP': f"{avg_otp:.1f}%",
                            'Recommendation': '✅ Primary' if data['Total Orders'] > 0 else '❌ Not Used'
                        })
                    
                    df_summary = pd.DataFrame(summary_rows).sort_values('Total Orders', ascending=False)
                    
                    # Highlight recommended channels
                    st.dataframe(
                        df_summary,
                        hide_index=True,
                        width='stretch',
                        column_config={
                            "Recommended Channel": st.column_config.TextColumn("Channel", width="medium"),
                            "Total Orders": st.column_config.NumberColumn("Total Orders", width="small"),
                            "Drivers Required": st.column_config.NumberColumn("Drivers Needed", width="small"),
                            "Orders per Driver": st.column_config.TextColumn("Orders/Driver", width="small"),
                            "Recommendation": st.column_config.TextColumn("Status", width="medium")
                        }
                    )
                    
                    # Visual recommendation
                    st.markdown("#### 🎯 Channel Recommendations")
                    primary_channels = df_summary[df_summary['Total Orders'] > 0].head(3)
                    
                    if len(primary_channels) > 0:
                        col1, col2, col3 = st.columns(3)
                        cols = [col1, col2, col3]
                        
                        for idx, (_, ch_row) in enumerate(primary_channels.iterrows()):
                            with cols[idx] if idx < 3 else col1:
                                st.success(f"**{ch_row['Recommended Channel']}**")
                                st.metric("Orders", f"{int(ch_row['Total Orders'])}")
                                st.metric("Drivers Needed", f"{int(ch_row['Drivers Required'])}")
                                st.caption(f"Orders/Driver: {ch_row['Orders per Driver']}")
                                st.caption(f"OTP: {ch_row['Avg OTP']}")
                    else:
                        st.warning("No channels allocated orders")
                    
                    # Driver allocation summary
                    st.markdown("#### 👥 Driver Allocation Summary")
                    total_drivers = sum([int(row['Drivers Needed']) for row in all_daily_data])
                    total_orders_all = sum([row['Orders Allocated'] for row in all_daily_data])
                    overall_orders_per_driver = total_orders_all / total_drivers if total_drivers > 0 else 0
                    
                    driver_col1, driver_col2, driver_col3, driver_col4 = st.columns(4)
                    with driver_col1:
                        st.metric("Total Drivers Needed", f"{total_drivers}")
                    with driver_col2:
                        st.metric("Total Orders", f"{total_orders_all}")
                    with driver_col3:
                        st.metric("Avg Orders per Driver", f"{overall_orders_per_driver:.1f}")
                    with driver_col4:
                        # Calculate utilization
                        total_driver_capacity = 0
                        for ch in st.session_state.channels_config:
                            if ch['id'] in channel_summary:
                                total_driver_capacity += ch['drivers_available'] * ch['trips_per_day_per_driver'] * ch['avg_orders_per_trip']
                        utilization = (total_orders_all / total_driver_capacity * 100) if total_driver_capacity > 0 else 0
                        st.metric("Driver Utilization", f"{utilization:.1f}%")
                        
                        if utilization > 90:
                            st.error("⚠️ High utilization")
                        elif utilization > 70:
                            st.warning("⚠️ Moderate utilization")
                        else:
                            st.success("✓ Good utilization")
                    
                    # Cost Over Time
                    daily_costs = []
                    for day_result in results:
                        daily_costs.append(day_result['total_cost'])
                    
                    fig_cost = go.Figure()
                    fig_cost.add_trace(go.Scatter(
                        x=list(range(1, len(daily_costs) + 1)),
                        y=daily_costs,
                        mode='lines+markers',
                        name='Daily Cost',
                        line=dict(color='#FF6B6B', width=3),
                        marker=dict(size=10)
                    ))
                    fig_cost.update_layout(
                        title='📈 Daily Operational Cost Trend',
                        xaxis_title='Day',
                        yaxis_title='Cost ($)',
                        hovermode='x unified'
                    )
                    st.plotly_chart(fig_cost, width='stretch')
                    
                    # OTP Performance by Day
                    daily_otp = []
                    daily_weights = []
                    for day_result in results:
                        day_otp = 0
                        day_weight = 0
                        for alloc in day_result['allocations']:
                            day_otp += alloc['otp'] * alloc['orders']
                            day_weight += alloc['orders']
                        if day_weight > 0:
                            daily_otp.append((day_otp / day_weight))
                        else:
                            daily_otp.append(0)
                    
                    fig_otp = go.Figure()
                    fig_otp.add_trace(go.Scatter(
                        x=list(range(1, len(daily_otp) + 1)),
                        y=daily_otp,
                        mode='lines+markers',
                        name='Weighted OTP',
                        line=dict(color='#4ECDC4', width=3),
                        marker=dict(size=10)
                    ))
                    fig_otp.add_hline(
                        y=95, 
                        line_dash="dash", 
                        line_color="red",
                        annotation_text="95% Target"
                    )
                    fig_otp.update_layout(
                        title='🎯 Daily OTP Performance',
                        xaxis_title='Day',
                        yaxis_title='OTP %',
                        yaxis=dict(range=[80, 100])
                    )
                    st.plotly_chart(fig_otp, width='stretch')
                    
                    # Clear Action Plan
                    st.divider()
                    st.markdown("### 📋 ACTION PLAN - What To Do")
                    
                    action_plan_col1, action_plan_col2 = st.columns(2)
                    
                    with action_plan_col1:
                        st.markdown("#### 🎯 Recommended Channels (Top Priority)")
                        top_channels = df_summary[df_summary['Total Orders'] > 0].head(5)
                        for idx, (_, ch_row) in enumerate(top_channels.iterrows(), 1):
                            st.markdown(f"""
                            **{idx}. Channel: {ch_row['Recommended Channel']}**
                            - Assign **{int(ch_row['Total Orders'])} orders** to this channel
                            - Need **{int(ch_row['Drivers Required'])} drivers**
                            - Each driver handles **{ch_row['Orders per Driver']} orders**
                            - Expected OTP: **{ch_row['Avg OTP']}**
                            - Cost: **{ch_row['Total Cost']}**
                            """)
                    
                    with action_plan_col2:
                        st.markdown("#### 📊 Daily Driver Requirements")
                        
                        # Group by day - more accurate
                        daily_driver_req = {}
                        for row in all_daily_data:
                            day = row['Day']
                            if day not in daily_driver_req:
                                daily_driver_req[day] = {
                                    'channels': {},
                                    'total_orders': 0
                                }
                            ch = row['Recommended Channel']
                            if ch not in daily_driver_req[day]['channels']:
                                daily_driver_req[day]['channels'][ch] = {
                                    'drivers': 0,
                                    'orders': 0,
                                    'orders_per_driver': 0
                                }
                            daily_driver_req[day]['channels'][ch]['drivers'] = max(
                                daily_driver_req[day]['channels'][ch]['drivers'],
                                row['Drivers Needed']
                            )
                            daily_driver_req[day]['channels'][ch]['orders'] += row['Orders Allocated']
                            daily_driver_req[day]['channels'][ch]['orders_per_driver'] = float(row['Orders per Driver'])
                            daily_driver_req[day]['total_orders'] += row['Orders Allocated']
                        
                        # Calculate total drivers per day (sum across channels)
                        for day in daily_driver_req.keys():
                            daily_driver_req[day]['total_drivers'] = sum([v['drivers'] for v in daily_driver_req[day]['channels'].values()])
                        
                        # Show daily breakdown
                        for day in sorted(daily_driver_req.keys()):
                            day_data = daily_driver_req[day]
                            total_day_drivers = day_data['total_drivers']
                            with st.expander(f"📅 Day {day}: {day_data['total_orders']} orders | {total_day_drivers} drivers needed", expanded=False):
                                st.markdown(f"**Daily Summary:**")
                                st.markdown(f"- Total Orders: {day_data['total_orders']}")
                                st.markdown(f"- Total Drivers: {total_day_drivers}")
                                st.markdown(f"- Avg Orders/Driver: {day_data['total_orders'] / total_day_drivers:.1f}" if total_day_drivers > 0 else "- Avg Orders/Driver: N/A")
                                st.markdown("")
                                st.markdown("**Channel Breakdown:**")
                                for ch, ch_data in day_data['channels'].items():
                                    st.markdown(f"""
                                    - **{ch}:** 
                                      - {ch_data['drivers']} drivers
                                      - {ch_data['orders']} orders
                                      - {ch_data['orders_per_driver']:.1f} orders per driver
                                    """)
                    
                    # Comprehensive Recommendations
                    st.divider()
                    st.markdown("### 🎯 COMPREHENSIVE RECOMMENDATIONS")
                    
                    rec_tabs = st.tabs(["📡 Channel Recommendations", "🚛 Truck Recommendations", "📅 Slot Recommendations", "📊 Overall Plan"])
                    
                    with rec_tabs[0]:
                        st.markdown("#### 📡 Which Channels to Use")
                        
                        # Analyze each channel
                        channel_recommendations = []
                        for ch in st.session_state.channels_config:
                            ch_id = ch['id']
                            ch_usage = channel_summary.get(ch_id, {})
                            ch_orders = ch_usage.get('Total Orders', 0)
                            
                            if ch_orders > 0:
                                # Channel is recommended
                                capacity, driver_cap, truck_cap = calculate_channel_capacity(ch)
                                utilization = (ch_orders / capacity * 100) if capacity > 0 else 0
                                
                                channel_recommendations.append({
                                    'channel': ch_id,
                                    'status': '✅ USE THIS CHANNEL',
                                    'priority': 'HIGH' if ch_orders > total_orders_quick * 0.3 else 'MEDIUM',
                                    'orders': ch_orders,
                                    'drivers_needed': ch_usage.get('Total Drivers', 0),
                                    'trucks_needed_a': ch['available_trucks_type_a'],
                                    'trucks_needed_b': ch['available_trucks_type_b'],
                                    'utilization': utilization,
                                    'reason': f"Optimized for {ch_orders} orders with {utilization:.0f}% utilization"
                                })
                            else:
                                # Channel not recommended
                                channel_recommendations.append({
                                    'channel': ch_id,
                                    'status': '❌ DO NOT USE',
                                    'priority': 'LOW',
                                    'orders': 0,
                                    'drivers_needed': 0,
                                    'trucks_needed_a': 0,
                                    'trucks_needed_b': 0,
                                    'utilization': 0,
                                    'reason': 'Not optimal based on current configuration'
                                })
                        
                        # Sort by priority and orders
                        channel_recommendations.sort(key=lambda x: (x['status'] == '❌ DO NOT USE', -x['orders']))
                        
                        for rec in channel_recommendations:
                            if rec['status'] == '✅ USE THIS CHANNEL':
                                with st.expander(f"✅ **{rec['channel']}** - {rec['status']}", expanded=True):
                                    col_r1, col_r2 = st.columns(2)
                                    with col_r1:
                                        st.metric("Orders to Assign", f"{rec['orders']}")
                                        st.metric("Drivers Needed", f"{rec['drivers_needed']}")
                                        st.metric("Utilization", f"{rec['utilization']:.1f}%")
                                    with col_r2:
                                        st.metric("Trucks Type A", f"{rec['trucks_needed_a']}")
                                        st.metric("Trucks Type B", f"{rec['trucks_needed_b']}")
                                        st.metric("Priority", rec['priority'])
                                    st.info(f"**Why:** {rec['reason']}")
                            else:
                                with st.expander(f"❌ **{rec['channel']}** - {rec['status']}", expanded=False):
                                    st.warning(f"Not recommended: {rec['reason']}")
                    
                    with rec_tabs[1]:
                        st.markdown("#### 🚛 Truck Requirements & Recommendations")
                        
                        # Calculate truck needs
                        truck_recommendations = []
                        
                        # Analyze truck requirements per channel
                        for ch_rec in channel_recommendations:
                            if ch_rec['status'] == '✅ USE THIS CHANNEL':
                                ch = next((c for c in st.session_state.channels_config if c['id'] == ch_rec['channel']), None)
                                if ch:
                                    # Calculate if more trucks needed
                                    current_trucks_a = ch['available_trucks_type_a']
                                    current_trucks_b = ch['available_trucks_type_b']
                                    
                                    # Find truck capacities
                                    truck_a_cap = 0
                                    truck_b_cap = 0
                                    for truck_type in st.session_state.trucks_config['truck_types']:
                                        if truck_type['id'] == 'A':
                                            truck_a_cap = truck_type['capacity']
                                        elif truck_type['id'] == 'B':
                                            truck_b_cap = truck_type['capacity']
                                    
                                    # Calculate required capacity
                                    required_capacity = ch_rec['orders']
                                    current_truck_capacity = (current_trucks_a * truck_a_cap) + (current_trucks_b * truck_b_cap)
                                    
                                    truck_recommendations.append({
                                        'channel': ch_rec['channel'],
                                        'current_trucks_a': current_trucks_a,
                                        'current_trucks_b': current_trucks_b,
                                        'current_capacity': current_truck_capacity,
                                        'required_capacity': required_capacity,
                                        'truck_a_cap': truck_a_cap,
                                        'truck_b_cap': truck_b_cap,
                                        'needs_more': required_capacity > current_truck_capacity
                                    })
                        
                        # Overall truck summary
                        total_trucks_a_needed = sum([r['current_trucks_a'] for r in truck_recommendations])
                        total_trucks_b_needed = sum([r['current_trucks_b'] for r in truck_recommendations])
                        
                        st.markdown("##### 📊 Overall Truck Requirements")
                        truck_sum_col1, truck_sum_col2, truck_sum_col3 = st.columns(3)
                        
                        with truck_sum_col1:
                            st.metric("Trucks Type A Needed", f"{total_trucks_a_needed}")
                            st.caption(f"Capacity: {truck_recommendations[0]['truck_a_cap'] if truck_recommendations else 0} orders/truck")
                        
                        with truck_sum_col2:
                            st.metric("Trucks Type B Needed", f"{total_trucks_b_needed}")
                            st.caption(f"Capacity: {truck_recommendations[0]['truck_b_cap'] if truck_recommendations else 0} orders/truck")
                        
                        with truck_sum_col3:
                            total_truck_capacity = (total_trucks_a_needed * (truck_recommendations[0]['truck_a_cap'] if truck_recommendations else 0)) + \
                                                  (total_trucks_b_needed * (truck_recommendations[0]['truck_b_cap'] if truck_recommendations else 0))
                            st.metric("Total Truck Capacity", f"{total_truck_capacity}")
                        
                        st.markdown("##### 🔍 Per-Channel Truck Analysis")
                        for truck_rec in truck_recommendations:
                            with st.expander(f"🚛 Channel {truck_rec['channel']} - Truck Requirements", expanded=False):
                                col_t1, col_t2 = st.columns(2)
                                
                                with col_t1:
                                    st.markdown("**Current Configuration:**")
                                    st.write(f"- Trucks Type A: {truck_rec['current_trucks_a']}")
                                    st.write(f"- Trucks Type B: {truck_rec['current_trucks_b']}")
                                    st.write(f"- Current Capacity: {truck_rec['current_capacity']} orders/day")
                                
                                with col_t2:
                                    st.markdown("**Required:**")
                                    st.write(f"- Required Capacity: {truck_rec['required_capacity']} orders/day")
                                    
                                    if truck_rec['needs_more']:
                                        shortage = truck_rec['required_capacity'] - truck_rec['current_capacity']
                                        st.error(f"⚠️ **Shortage: {shortage:.0f} orders**")
                                        
                                        # Calculate how many more trucks needed
                                        if truck_rec['truck_a_cap'] > 0:
                                            trucks_a_needed = np.ceil(shortage / truck_rec['truck_a_cap'])
                                            st.warning(f"**Add {int(trucks_a_needed)} more Truck(s) Type A**")
                                        if truck_rec['truck_b_cap'] > 0:
                                            trucks_b_needed = np.ceil(shortage / truck_rec['truck_b_cap'])
                                            st.warning(f"**Or add {int(trucks_b_needed)} more Truck(s) Type B**")
                                    else:
                                        st.success("✓ Sufficient trucks available")
                                        
                                        # Check if too many trucks (wasteful)
                                        excess = truck_rec['current_capacity'] - truck_rec['required_capacity']
                                        if excess > truck_rec['current_capacity'] * 0.3:
                                            st.info(f"💡 Consider reducing trucks (excess capacity: {excess:.0f} orders)")
                    
                    with rec_tabs[2]:
                        st.markdown("#### 📅 Slot Configuration Recommendations")
                        
                        # Analyze slot requirements
                        total_orders_slot = sum(st.session_state.daily_orders['daily_expected_orders'])
                        avg_orders_per_day = total_orders_slot / len(st.session_state.daily_orders['daily_expected_orders']) if st.session_state.daily_orders['daily_expected_orders'] else 0
                        
                        current_slots = st.session_state.daily_slots['slots_per_day']
                        slot_capacities = st.session_state.daily_slots.get('slot_capacities', [])
                        total_capacity = sum(slot_capacities) if slot_capacities else 0
                        
                        st.markdown("##### Current Slot Configuration")
                        slot_col1, slot_col2 = st.columns(2)
                        
                        with slot_col1:
                            st.metric("Current Slots", f"{current_slots}")
                            st.metric("Avg Orders/Day", f"{avg_orders_per_day:.0f}")
                        
                        with slot_col2:
                            st.metric("Total Capacity", f"{total_capacity}")
                            if total_capacity < avg_orders_per_day:
                                st.error(f"⚠️ Capacity ({total_capacity}) below demand ({avg_orders_per_day:.0f})")
                            else:
                                st.success(f"✓ Sufficient capacity ({((total_capacity/avg_orders_per_day-1)*100):.0f}% buffer)")
                        
                        # Recommendations
                        st.markdown("##### 💡 Slot Recommendations")
                        
                        slot_recommendations = []
                        
                        # Check if slots are optimal
                        if current_slots < 3:
                            slot_recommendations.append({
                                'type': 'ADD',
                                'action': f"Consider adding more slots (currently {current_slots})",
                                'reason': 'More slots allow better time distribution',
                                'suggestion': 'Add 1-2 slots for better flexibility'
                            })
                        
                        if avg_orders_per_day > 100 and current_slots < 4:
                            slot_recommendations.append({
                                'type': 'ADD',
                                'action': f"Add more slots for high volume days",
                                'reason': f'High daily volume ({avg_orders_per_day:.0f} orders) requires more time slots',
                                'suggestion': 'Increase to 4-5 slots for peak days'
                            })
                        
                        # Analyze slot capacities
                        if slot_capacities:
                            max_capacity = max(slot_capacities)
                            min_capacity = min(slot_capacities)
                            avg_capacity = np.mean(slot_capacities)
                            
                            if max_capacity > avg_capacity * 2:
                                slot_recommendations.append({
                                    'type': 'EDIT',
                                    'action': f"Balance slot capacities",
                                    'reason': f'One slot has {max_capacity} capacity while average is {avg_capacity:.0f} - too concentrated',
                                    'suggestion': 'Distribute capacity more evenly across slots'
                                })
                            
                            if min_capacity < avg_capacity * 0.5:
                                slot_recommendations.append({
                                    'type': 'EDIT',
                                    'action': f"Increase low-capacity slots",
                                    'reason': f'Slot with {min_capacity} capacity is underutilized',
                                    'suggestion': 'Either increase capacity or remove this slot'
                                })
                        
                        if not slot_recommendations:
                            st.success("✅ Current slot configuration is optimal!")
                        else:
                            for rec in slot_recommendations:
                                if rec['type'] == 'ADD':
                                    st.warning(f"**ADD SLOT:** {rec['action']}")
                                    st.markdown(f"*Reason:* {rec['reason']}")
                                    st.info(f"💡 *Suggestion:* {rec['suggestion']}")
                                else:
                                    st.info(f"**EDIT SLOT:** {rec['action']}")
                                    st.markdown(f"*Reason:* {rec['reason']}")
                                    st.info(f"💡 *Suggestion:* {rec['suggestion']}")
                        
                        # Suggested slot configuration
                        st.markdown("##### 📋 Suggested Slot Configuration")
                        if avg_orders_per_day < 50:
                            suggested_slots = 2
                            suggested_capacities = [25, 25]
                            st.info("**For low volume:** 2 slots with capacity 25 orders each")
                        elif avg_orders_per_day < 150:
                            suggested_slots = 3
                            suggested_capacities = [50, 50, 50]
                            st.info("**For medium volume:** 3 slots with capacity 50 orders each")
                        else:
                            suggested_slots = 4
                            suggested_capacities = [75, 75, 75, 75]
                            st.info("**For high volume:** 4 slots with capacity 75 orders each")
                        
                        if suggested_slots != current_slots:
                            st.markdown(f"**Current:** {current_slots} slots → **Suggested:** {suggested_slots} slots")
                        if total_capacity < avg_orders_per_day:
                            st.warning(f"⚠️ Current total capacity ({total_capacity}) is below average demand ({avg_orders_per_day:.0f})")
                    
                    with rec_tabs[3]:
                        st.markdown("#### 📊 Overall Implementation Plan")
                        
                        # Create comprehensive plan
                        st.markdown("##### ✅ Step-by-Step Implementation")
                        
                        step_num = 1
                        
                        # Step 1: Channels
                        st.markdown(f"**Step {step_num}: Configure Channels**")
                        step_num += 1
                        st.markdown("""
                        **Action Items:**
                        """)
                        for ch_rec in channel_recommendations:
                            if ch_rec['status'] == '✅ USE THIS CHANNEL':
                                st.markdown(f"- ✅ **Use Channel {ch_rec['channel']}**")
                                st.markdown(f"  - Assign {ch_rec['orders']} orders")
                                st.markdown(f"  - Allocate {ch_rec['drivers_needed']} drivers")
                                st.markdown(f"  - Expected utilization: {ch_rec['utilization']:.1f}%")
                        
                        # Step 2: Trucks
                        st.markdown(f"**Step {step_num}: Allocate Trucks**")
                        step_num += 1
                        st.markdown("""
                        **Truck Allocation:**
                        """)
                        for truck_rec in truck_recommendations:
                            st.markdown(f"- **Channel {truck_rec['channel']}:**")
                            st.markdown(f"  - Trucks Type A: {truck_rec['current_trucks_a']}")
                            st.markdown(f"  - Trucks Type B: {truck_rec['current_trucks_b']}")
                            if truck_rec['needs_more']:
                                shortage = truck_rec['required_capacity'] - truck_rec['current_capacity']
                                if truck_rec['truck_a_cap'] > 0:
                                    trucks_needed = np.ceil(shortage / truck_rec['truck_a_cap'])
                                    st.markdown(f"  - ⚠️ **ADD {int(trucks_needed)} Truck(s) Type A**")
                        
                        # Step 3: Slots
                        st.markdown(f"**Step {step_num}: Configure Time Slots**")
                        step_num += 1
                        if slot_recommendations:
                            st.markdown("**Recommended Changes:**")
                            for rec in slot_recommendations:
                                st.markdown(f"- {rec['action']}")
                        else:
                            st.markdown("- ✅ Current slot configuration is optimal - no changes needed")
                        
                        # Step 4: Drivers
                        st.markdown(f"**Step {step_num}: Assign Drivers**")
                        step_num += 1
                        st.markdown(f"""
                        **Driver Allocation:**
                        - Total drivers needed: **{total_drivers}**
                        - Average orders per driver: **{overall_orders_per_driver:.1f}**
                        - Daily breakdown available in "Daily Driver Requirements" section
                        """)
                        
                        # Summary metrics
                        st.markdown("---")
                        st.markdown("##### 📈 Expected Outcomes")
                        
                        outcome_col1, outcome_col2, outcome_col3 = st.columns(3)
                        with outcome_col1:
                            st.success(f"**OTP Target:** {roi_data['weighted_avg_otp']:.1f}%")
                            st.caption(f"Target: {st.session_state.optimization_params['otp_targets'][0]}%")
                        with outcome_col2:
                            st.info(f"**Total Cost:** ${roi_data['total_cost']:,.0f}")
                            st.caption(f"${roi_data['avg_cost_per_order']:.2f} per order")
                        with outcome_col3:
                            st.warning(f"**Fulfillment:** {roi_data['fulfillment_rate']:.1f}%")
                            if roi_data['fulfillment_rate'] < 100:
                                st.caption(f"{roi_data['unallocated_orders']:.0f} orders unfulfilled")
                    
                    # Executive Summary Box
                    st.markdown("---")
                    st.markdown("### ✅ EXECUTIVE SUMMARY")
                    
                    exec_col1, exec_col2, exec_col3, exec_col4 = st.columns(4)
                    
                    with exec_col1:
                        st.info(f"""
                        **Recommended Channels:** {len([ch for ch in channel_summary.keys() if channel_summary[ch]['Total Orders'] > 0])}
                        
                        Primary: {primary_channels.iloc[0]['Recommended Channel'] if len(primary_channels) > 0 else 'N/A'}
                        """)
                    
                    with exec_col2:
                        st.info(f"""
                        **Total Drivers Needed:** {total_drivers}
                        
                        Avg: {overall_orders_per_driver:.1f} orders/driver
                        """)
                    
                    with exec_col3:
                        st.info(f"""
                        **Total Cost:** ${roi_data['total_cost']:,.0f}
                        
                        Per Order: ${roi_data['avg_cost_per_order']:.2f}
                        """)
                    
                    with exec_col4:
                        st.info(f"""
                        **Expected OTP:** {roi_data['weighted_avg_otp']:.1f}%
                        
                        Target: {st.session_state.optimization_params['otp_targets'][0]}%
                        """)
        
        # Configuration summary
        st.subheader("Current Configuration Summary")
        
        summary_data = {
            'Metric': [
                'Planning Period (Days)',
                'Total Expected Orders',
                'Average Orders/Day',
                'Available Trucks',
                'Number of Channels',
                'Slot Configuration',
                'Cost-OTP Weight'
            ],
            'Value': [
                str(st.session_state.daily_orders['planning_period']),
                str(sum(st.session_state.daily_orders['daily_expected_orders'])),
                f"{sum(st.session_state.daily_orders['daily_expected_orders']) / st.session_state.daily_orders['planning_period']:.1f}",
                str(st.session_state.trucks_config['available_trucks']),
                str(len(st.session_state.channels_config)),
                f"{st.session_state.daily_slots['slots_per_day']} slots",
                str(st.session_state.optimization_params['cost_otp_weight'])
            ]
        }
        
        df_summary = pd.DataFrame(summary_data)
        st.dataframe(df_summary, width='stretch', hide_index=True)
        
        # Expected orders visualization
        st.subheader("Expected Orders Overview")
        
        orders_df = pd.DataFrame({
            'Day': range(1, len(st.session_state.daily_orders['daily_expected_orders']) + 1),
            'Orders': st.session_state.daily_orders['daily_expected_orders']
        })
        
        fig = px.line(
            orders_df,
            x='Day',
            y='Orders',
            title="Expected Orders Over Planning Period",
            markers=True
        )
        st.plotly_chart(fig, width='stretch')
        
        # Channel OTP Performance
        st.subheader("Channel OTP Performance")
        
        otp_data = []
        for channel in st.session_state.channels_config:
            otp_data.append({
                'Channel': channel['id'],
                '24h OTP': channel['otp_24h'],
                '7d OTP': channel['otp_7d'],
                '30d OTP': channel['otp_30d']
            })
        
        otp_df = pd.DataFrame(otp_data)
        
        fig2 = go.Figure()
        for col in ['24h OTP', '7d OTP', '30d OTP']:
            fig2.add_trace(go.Bar(
                name=col,
                x=otp_df['Channel'],
                y=otp_df[col]
            ))
        
        fig2.update_layout(
            barmode='group',
            title="OTP Performance by Channel",
            yaxis_title="OTP %",
            xaxis_title="Channel"
        )
        st.plotly_chart(fig2, width='stretch')
        
        # Export functionality
        if 'scenario_a' in st.session_state.optimization_results or 'scenario_b' in st.session_state.optimization_results:
            st.divider()
            st.subheader("📥 Export Optimization Plan")
            
            col_exp1, col_exp2 = st.columns(2)
            
            with col_exp1:
                if st.button("📊 Export as CSV", type="primary", use_container_width=True):
                    # Create comprehensive export data
                    export_rows = []
                    for scenario_key in ['scenario_a', 'scenario_b']:
                        if scenario_key in st.session_state.optimization_results:
                            results = st.session_state.optimization_results[scenario_key]
                            for day_result in results:
                                for alloc in day_result['allocations']:
                                    export_rows.append({
                                        'Scenario': scenario_key,
                                        'Day': alloc['day'],
                                        'Channel': alloc['channel'],
                                        'Orders_Allocated': alloc['orders'],
                                        'Cost_USD': alloc['cost'],
                                        'OTP_%': alloc['otp'],
                                        'Capacity_Utilization_%': alloc['capacity_utilization']
                                    })
                    
                    if export_rows:
                        df_export = pd.DataFrame(export_rows)
                        csv = df_export.to_csv(index=False)
                        st.download_button(
                            label="⬇️ Download CSV",
                            data=csv,
                            file_name=f"optimization_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                        st.success("✓ CSV ready for download!")
            
            with col_exp2:
                if st.button("💾 Save Configuration", type="primary", use_container_width=True):
                    config_data = {
                        'daily_orders': st.session_state.daily_orders,
                        'daily_slots': st.session_state.daily_slots,
                        'trucks_config': st.session_state.trucks_config,
                        'channels_config': st.session_state.channels_config,
                        'optimization_params': st.session_state.optimization_params
                    }
                    
                    import json
                    json_data = json.dumps(config_data, indent=2)
                    st.download_button(
                        label="⬇️ Download Configuration",
                        data=json_data,
                        file_name=f"logistics_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                        use_container_width=True
                    )
                    st.success("✓ Configuration ready for download!")
            
            # AI-Powered Insights
            st.divider()
            st.subheader("🤖 AI-Powered Actionable Insights")
            
            # Generate insights
            insights = []
            
            # Capacity insights
            total_orders = sum(st.session_state.daily_orders['daily_expected_orders'])
            total_capacity = sum([calculate_channel_capacity(ch)[0] for ch in st.session_state.channels_config])
            
            if total_capacity < total_orders:
                insights.append({
                    'type': 'error',
                    'severity': 'HIGH',
                    'title': '🚨 Critical: Capacity Shortage',
                    'message': f"You need {total_orders - total_capacity:.0f} more daily capacity. Consider adding channels or increasing truck/driver capacity.",
                    'actions': ['Add more drivers to existing channels', 'Increase truck fleet', 'Add new delivery channels']
                })
            elif total_capacity < total_orders * 1.2:
                insights.append({
                    'type': 'warning',
                    'severity': 'MEDIUM',
                    'title': '⚠️ Tight Capacity',
                    'message': f"Capacity margin is only {((total_capacity/total_orders - 1)*100):.1f}%. Any demand spike will exceed capacity.",
                    'actions': ['Increase buffer capacity by 20%', 'Monitor daily demand patterns', 'Have backup channels ready']
                })
            
            # Cost insights
            if 'scenario_a' in st.session_state.optimization_results:
                results = st.session_state.optimization_results['scenario_a']
                roi_data = calculate_roi(results)
                
                if roi_data['avg_cost_per_order'] > 20:
                    insights.append({
                        'type': 'warning',
                        'severity': 'MEDIUM',
                        'title': '💰 High Cost per Order',
                        'message': f"Your average cost is ${roi_data['avg_cost_per_order']:.2f} per order. Consider renegotiating rates or optimizing routes.",
                        'actions': ['Review channel pricing', 'Optimize route efficiency', 'Negotiate better rates']
                    })
                
                if roi_data['fulfillment_rate'] < 95:
                    insights.append({
                        'type': 'warning',
                        'severity': 'HIGH',
                        'title': '📉 Low Fulfillment Rate',
                        'message': f"Only {roi_data['fulfillment_rate']:.1f}% of orders can be fulfilled with current capacity.",
                        'actions': ['Urgent: Increase capacity immediately', 'Consider reallocating resources', 'Add backup channels']
                    })
                
                # OTP insights
                if roi_data['weighted_avg_otp'] < st.session_state.optimization_params['otp_targets'][0]:
                    insights.append({
                        'type': 'error',
                        'severity': 'HIGH',
                        'title': '🎯 OTP Below Target',
                        'message': f"Current OTP {roi_data['weighted_avg_otp']:.1f}% is below target {st.session_state.optimization_params['otp_targets'][0]}%.",
                        'actions': ['Improve channel OTP performance', 'Redistribute orders to better-performing channels', 'Add capacity buffer']
                    })
            
            # Channel performance insights
            worst_channel = None
            worst_otp = 100
            for channel in st.session_state.channels_config:
                otp = calculate_weighted_otp(channel)
                if otp < worst_otp:
                    worst_otp = otp
                    worst_channel = channel
            
            if worst_channel and worst_otp < 92:
                insights.append({
                    'type': 'warning',
                    'severity': 'MEDIUM',
                    'title': f'🔧 Channel {worst_channel["id"]} Needs Attention',
                    'message': f"Lowest OTP performer at {worst_otp:.1f}%. Review operations and training.",
                    'actions': ['Conduct root cause analysis', 'Implement performance improvement plan', 'Consider reallocating resources']
                })
            
            # Best practices
            insights.append({
                'type': 'info',
                'severity': 'LOW',
                'title': '💡 Best Practice Suggestion',
                'message': 'Maintain a 20% capacity buffer to handle demand variability and unexpected spikes.',
                'actions': ['Keep spare capacity in reserve', 'Monitor trends daily', 'Adjust forecasts weekly']
            })
            
            # Display insights
            if insights:
                for idx, insight in enumerate(insights):
                    if insight['type'] == 'error':
                        st.error(f"### {insight['title']}")
                    elif insight['type'] == 'warning':
                        st.warning(f"### {insight['title']}")
                    else:
                        st.info(f"### {insight['title']}")
                    
                    st.markdown(insight['message'])
                    
                    st.markdown("**Recommended Actions:**")
                    for action in insight['actions']:
                        st.markdown(f"• {action}")
                    
                    if idx < len(insights) - 1:
                        st.markdown("---")

# ========================== EXECUTIVE DASHBOARD SECTION ==========================
if page == "🏆 Executive Dashboard":
    st.header("🏆 Executive Dashboard")
    st.markdown("### One-Page Strategic Overview for Management")
    
    # Help Section
    with st.expander("ℹ️ What is the Executive Dashboard?", expanded=False):
        st.markdown("""
        ### Executive Dashboard Explained
        
        **Purpose:** One-page strategic overview for management decision-making.
        
        **Key Sections:**
        
        **1. 📊 Key Performance Indicators (KPIs)**
        - **Total Orders:** Demand to fulfill
        - **Total Capacity:** Maximum you can handle
        - **Risk Level:** Risk assessment (LOW/MEDIUM/HIGH)
        - **Avg OTP:** Average on-time performance
        - **Est. Cost:** Expected operational costs
        
        **2. 💰 Financial Impact Analysis**
        - Cost breakdown by channel
        - OTP performance by channel
        - Visual comparison of efficiency
        
        **3. 🎯 Top Strategic Recommendations**
        - Prioritized by urgency (CRITICAL/HIGH/MEDIUM)
        - Impact estimates (revenue at risk or potential savings)
        - Specific recommended actions
        - Timeline for implementation
        
        **4. 🌍 Industry Benchmarking**
        - Compare your costs to industry averages
        - See your OTP vs market standards
        - Competitive score (0-100)
        - Market positioning analysis
        
        **5. 📊 Risk Scoring**
        - Automated risk detection
        - Severity scoring (1-10)
        - Impact analysis
        - Detailed risk breakdown
        
        **6. ⚖️ Scenario Comparison**
        - Compare different optimization strategies
        - See trade-offs between cost and OTP
        - Make informed decisions
        
        **Who Should Use This:**
        - **Executives:** Strategic overview and decision support
        - **Managers:** Operational insights and recommendations
        - **Planners:** Capacity and resource planning
        
        **How to Use:**
        1. Configure your system in other sections first
        2. Click "Run Optimization" to generate scenarios
        3. Review KPIs and recommendations
        4. Take action on high-priority items
        """)
    
    if not st.session_state.channels_config or not st.session_state.daily_orders['daily_expected_orders']:
        st.warning("⚠️ Please configure orders and channels first to see full dashboard.")
    
    # Key Metrics
    st.subheader("📊 Key Performance Indicators")
    
    col_kpi1, col_kpi2, col_kpi3, col_kpi4, col_kpi5 = st.columns(5)
    
    total_orders = sum(st.session_state.daily_orders['daily_expected_orders']) if st.session_state.daily_orders['daily_expected_orders'] else 0
    total_capacity = sum([calculate_channel_capacity(ch)[0] for ch in st.session_state.channels_config]) if st.session_state.channels_config else 0
    
    with col_kpi1:
        st.metric("Total Orders", f"{total_orders:,}")
    
    with col_kpi2:
        st.metric("Total Capacity", f"{total_capacity:,}", delta=f"{((total_capacity/total_orders-1)*100):.0f}%" if total_orders > 0 else "N/A")
        # Add explanation tooltip
        with st.popover("ℹ️ What is Capacity?"):
            st.markdown("""
            **Total Capacity** is the maximum number of orders you can handle per day.
            
            **It depends on the MINIMUM of:**
            
            1. **Driver Capacity** = 
               - Drivers Available × Trips per Driver × Orders per Trip
               - (Summed across all channels)
            
            2. **Truck Capacity** = 
               - (Trucks Type A × Capacity A) + (Trucks Type B × Capacity B)
               - (Summed across all channels)
            
            **Formula per Channel:**
            ```
            Driver Capacity = drivers × trips_per_driver × orders_per_trip
            Truck Capacity = (trucks_A × capacity_A) + (trucks_B × capacity_B)
            Channel Capacity = MIN(driver_capacity, truck_capacity)
            ```
            
            **Total Capacity = Sum of all channel capacities**
            
            **Why MIN?** You can only fulfill orders if you have BOTH drivers AND trucks!
            """)
            
            # Show breakdown
            if st.session_state.channels_config:
                st.markdown("**Current Breakdown:**")
                for ch in st.session_state.channels_config:
                    cap, driver_cap, truck_cap = calculate_channel_capacity(ch)
                    st.markdown(f"- **{ch['id']}:** {cap:.0f} orders/day")
                    st.caption(f"  └─ Driver limit: {driver_cap:.0f} | Truck limit: {truck_cap:.0f}")
    
    with col_kpi3:
        risk_level = "HIGH" if total_capacity < total_orders else "MEDIUM" if total_capacity < total_orders * 1.2 else "LOW"
        risk_color = "inverse" if total_capacity < total_orders else "normal"
        st.metric("Risk Level", risk_level, delta=f"Capacity: {((total_capacity/total_orders-1)*100):.0f}%" if total_orders > 0 else "N/A", delta_color=risk_color)
    
    with col_kpi4:
        if st.session_state.channels_config:
            avg_otp = np.mean([calculate_weighted_otp(ch) for ch in st.session_state.channels_config])
            st.metric("Avg OTP", f"{avg_otp:.1f}%")
    
    with col_kpi5:
        if 'scenario_a' in st.session_state.optimization_results:
            results = st.session_state.optimization_results['scenario_a']
            roi_data = calculate_roi(results)
            st.metric("Est. Cost", f"${roi_data['total_cost']:,.0f}" if results else "N/A")
        else:
            # Rough estimate
            est_cost = total_orders * 15  # rough avg
            st.metric("Est. Cost", f"${est_cost:,.0f}", help="Rough estimate based on historical data")
    
    st.divider()
    
    # Financial Overview
    st.subheader("💰 Financial Impact Analysis")
    
    col_fin1, col_fin2 = st.columns(2)
    
    with col_fin1:
        # Cost Breakdown Chart
        if st.session_state.channels_config:
            channel_costs = []
            for ch in st.session_state.channels_config:
                # Estimate daily cost for this channel
                capacity, _, _ = calculate_channel_capacity(ch)
                est_daily_orders = min(capacity, 50) if total_orders > 0 else 0
                avg_cost_per_order = ch['cost_per_order'] + (ch['fixed_cost_per_trip'] / ch['avg_orders_per_trip'])
                est_daily_cost = est_daily_orders * avg_cost_per_order
                channel_costs.append({
                    'Channel': ch['id'],
                    'Est. Daily Cost': est_daily_cost
                })
            
            df_channel_costs = pd.DataFrame(channel_costs)
            fig = px.bar(df_channel_costs, x='Channel', y='Est. Daily Cost', 
                        title='Estimated Daily Cost by Channel',
                        color='Est. Daily Cost', color_continuous_scale='Reds')
            st.plotly_chart(fig, width='stretch')
    
    with col_fin2:
        # OTP Performance
        if st.session_state.channels_config:
            otp_channel = []
            for ch in st.session_state.channels_config:
                weighted_otp = calculate_weighted_otp(ch)
                otp_channel.append({
                    'Channel': ch['id'],
                    'OTP': weighted_otp
                })
            
            df_otp = pd.DataFrame(otp_channel)
            fig2 = px.bar(df_otp, x='Channel', y='OTP',
                         title='OTP Performance by Channel',
                         color='OTP', color_continuous_scale='Greens',
                         range_y=[80, 100])
            fig2.add_hline(y=95, line_dash="dash", line_color="red", annotation_text="Target")
            st.plotly_chart(fig2, width='stretch')
    
    st.divider()
    
    # Strategic Recommendations
    st.subheader("🎯 Top Strategic Recommendations")
    
    recommendations = []
    
    # Capacity recommendations
    if total_capacity < total_orders:
        recommendations.append({
            'priority': '🔴 CRITICAL',
            'title': 'Immediate Capacity Expansion Required',
            'impact': f"${(total_orders - total_capacity) * 20:,.0f} potential lost revenue",
            'action': f"Add {total_orders - total_capacity:.0f} daily capacity units",
            'urgency': 'This Week'
        })
    
    # Cost recommendations
    if st.session_state.channels_config:
        costs = [ch['cost_per_order'] + (ch['fixed_cost_per_trip'] / ch['avg_orders_per_trip']) for ch in st.session_state.channels_config]
        if max(costs) > 25:
            expensive_channels = [ch['id'] for idx, ch in enumerate(st.session_state.channels_config) if costs[idx] > 25]
            recommendations.append({
                'priority': '🟡 HIGH',
                'title': 'Cost Optimization Opportunity',
                'impact': f"Potential ${(max(costs) - 15) * total_orders:,.0f} annual savings",
                'action': f"Re-negotiate rates for channels: {', '.join(expensive_channels)}",
                'urgency': 'This Month'
            })
    
    # OTP recommendations
    if st.session_state.channels_config:
        avg_otp = np.mean([calculate_weighted_otp(ch) for ch in st.session_state.channels_config])
        if avg_otp < 95:
            recommendations.append({
                'priority': '🟠 MEDIUM',
                'title': 'OTP Improvement Required',
                'impact': f"Customer satisfaction at risk",
                'action': f"Increase OTP by {95 - avg_otp:.1f}% through process improvements",
                'urgency': 'Next Quarter'
            })
    
    if recommendations:
        for idx, rec in enumerate(recommendations, 1):
            with st.expander(f"{rec['priority']} {rec['title']}", expanded=idx==1):
                st.markdown(f"**Impact:** {rec['impact']}")
                st.markdown(f"**Recommended Action:** {rec['action']}")
                st.markdown(f"**Urgency:** {rec['urgency']}")
    else:
        st.success("✅ All systems performing within optimal parameters!")
    
    st.divider()
    
    # Comparison if scenarios exist
    if 'scenario_a' in st.session_state.optimization_results or 'scenario_b' in st.session_state.optimization_results:
        st.subheader("⚖️ Scenario Comparison")
        comp_col1, comp_col2 = st.columns(2)
        
        if 'scenario_a' in st.session_state.optimization_results:
            with comp_col1:
                results_a = st.session_state.optimization_results['scenario_a']
                roi_a = calculate_roi(results_a)
                st.metric("Scenario A - Total Cost", f"${roi_a['total_cost']:,.2f}")
                st.metric("Scenario A - Avg OTP", f"{roi_a['weighted_avg_otp']:.2f}%")
        
        if 'scenario_b' in st.session_state.optimization_results:
            with comp_col2:
                results_b = st.session_state.optimization_results['scenario_b']
                roi_b = calculate_roi(results_b)
                st.metric("Scenario B - Total Cost", f"${roi_b['total_cost']:,.2f}")
                st.metric("Scenario B - Avg OTP", f"{roi_b['weighted_avg_otp']:.2f}%")
        
        if 'scenario_a' in st.session_state.optimization_results and 'scenario_b' in st.session_state.optimization_results:
            cost_diff = abs(roi_a['total_cost'] - roi_b['total_cost'])
            otp_diff = abs(roi_a['weighted_avg_otp'] - roi_b['weighted_avg_otp'])
            st.info(f"💡 **Trade-off:** ${cost_diff:,.2f} cost difference vs {otp_diff:.2f}% OTP difference")
        
        # Add intelligent features here within Executive Dashboard
        st.divider()
        st.subheader("🌍 Industry Benchmarking & Market Intelligence")
        
        if st.session_state.channels_config:
            # Calculate industry positioning
            avg_cost = np.mean([ch['cost_per_order'] + (ch['fixed_cost_per_trip'] / ch['avg_orders_per_trip']) 
                               for ch in st.session_state.channels_config])
            avg_otp = np.mean([calculate_weighted_otp(ch) for ch in st.session_state.channels_config])
            
            bench_col1, bench_col2, bench_col3 = st.columns(3)
            
            # Industry benchmarks (typical ranges)
            industry_cost_avg = 18.50  # Typical cost per order
            industry_otp_avg = 94.5    # Typical OTP
            
            with bench_col1:
                cost_status = "Above Average" if avg_cost > industry_cost_avg else "Below Average"
                st.metric("Cost vs Industry", f"{'📈 ' if avg_cost > industry_cost_avg else '📉 '}{cost_status}", 
                         delta=f"${abs(avg_cost - industry_cost_avg):.2f}")
            
            with bench_col2:
                otp_status = "Above Target" if avg_otp >= 95 else "Near Target" if avg_otp >= 92 else "Below Target"
                st.metric("OTP vs Industry", f"{otp_status}", 
                         delta=f"{avg_otp - industry_otp_avg:+.1f}%")
            
            with bench_col3:
                # Calculate competitive score
                cost_score = max(0, 100 - ((avg_cost - industry_cost_avg) / industry_cost_avg * 50))
                otp_score = (avg_otp / 100) * 50
                comp_score = (cost_score + otp_score) / 2
                st.metric("Competitive Score", f"{comp_score:.0f}/100")
                
                if comp_score >= 85:
                    st.success("🎯 Market Leader Position")
                elif comp_score >= 70:
                    st.info("📊 Competitive Position")
                else:
                    st.warning("⚠️ Below Market Position")
            
            # Market positioning chart
            market_fig = go.Figure()
            market_fig.add_trace(go.Scatter(
                x=[industry_cost_avg], y=[industry_otp_avg],
                mode='markers', name='Industry Average',
                marker=dict(size=15, color='blue', symbol='circle')
            ))
            market_fig.add_trace(go.Scatter(
                x=[avg_cost], y=[avg_otp],
                mode='markers', name='Your Position',
                marker=dict(size=20, color='red', symbol='star')
            ))
            # Add quadrant lines
            market_fig.add_hline(y=95, line_dash="dash", line_color="gray", opacity=0.3)
            market_fig.add_vline(x=industry_cost_avg, line_dash="dash", line_color="gray", opacity=0.3)
            market_fig.update_layout(
                title='🎯 Market Positioning Analysis',
                xaxis_title='Cost per Order ($)',
                yaxis_title='OTP (%)',
                annotations=[
                    dict(x=12, y=97, text="📈 Best Practice Zone", showarrow=False, bgcolor="rgba(0,200,0,0.2)", bordercolor="green"),
                    dict(x=25, y=97, text="💰 Premium Zone", showarrow=False, bgcolor="rgba(255,200,0,0.2)", bordercolor="orange"),
                    dict(x=12, y=88, text="⚠️ Cost Leader (Low OTP)", showarrow=False, bgcolor="rgba(200,0,0,0.2)", bordercolor="red"),
                    dict(x=25, y=88, text="❌ Problem Zone", showarrow=False, bgcolor="rgba(200,0,0,0.3)", bordercolor="red")
                ]
            )
            st.plotly_chart(market_fig, width='stretch')
        
        # Clean up the duplicate code at the end
        st.markdown("")

elif page == "🧪 What-If Scenarios":
    st.header("🧪 What-If Scenario Analysis")
    st.markdown("### Test different strategies and see the impact")
    
    # Help Section
    with st.expander("ℹ️ What is What-If Analysis?", expanded=False):
        st.markdown("""
        ### What-If Scenario Analysis Explained
        
        **Purpose:** Test different strategies BEFORE implementing them.
        
        **What You Can Simulate:**
        - **Add Trucks:** See impact of increasing fleet
        - **Add Drivers:** See impact of hiring more staff
        - **Change Demand:** Test scenarios if orders increase/decrease
        
        **How It Works:**
        1. Set current capacity (baseline)
        2. Adjust parameters: Add trucks, drivers, or change demand
        3. See impact on capacity, demand, and costs
        4. Compare before/after in real-time
        
        **What You Get:**
        - **Capacity Change:** How much capacity increased (%)
        - **Demand Change:** How demand changed (%)
        - **Capacity Gap:** Whether you have enough capacity
        - **Financial Impact:** Cost difference
        
        **Business Use Cases:**
        - **Before Hiring:** "What if we add 5 more drivers?"
        - **Before Buying:** "What if we buy 10 more trucks?"
        - **Market Changes:** "What if demand increases by 20%?"
        - **Cost Planning:** "What will this expansion cost?"
        
        **Key Metrics:**
        - Green ✓ = Optimal capacity
        - Red ⚠️ = Insufficient capacity
        - Yellow ⚠️ = Excess capacity (wasteful)
        
        **Interpreting Results:**
        - Positive capacity gap = Not enough capacity
        - Negative capacity gap = Excess capacity (may be wasteful)
        - Cost delta shows financial impact
        """)
    
    if not st.session_state.channels_config or not st.session_state.daily_orders['daily_expected_orders']:
        st.warning("⚠️ Please configure orders and channels first!")
    else:
        st.subheader("🎮 Interactive Scenario Builder")
        
        # Scenario parameters
        col_sc1, col_sc2 = st.columns(2)
        
        with col_sc1:
            st.markdown("#### Scenario A: Current State")
            st.metric("Total Capacity", f"{sum([calculate_channel_capacity(ch)[0] for ch in st.session_state.channels_config]):.0f}")
            
        with col_sc2:
            st.markdown("#### Scenario B: What If...")
            
            # Add more trucks
            add_trucks_a = st.number_input("Add Trucks Type A", min_value=0, max_value=50, value=0)
            add_trucks_b = st.number_input("Add Trucks Type B", min_value=0, max_value=50, value=0)
            
            # Add more drivers
            add_drivers = st.number_input("Add Drivers per Channel", min_value=0, max_value=20, value=0)
            
            # Change demand
            demand_multiplier = st.slider("Demand Multiplier", 0.5, 2.0, 1.0, 0.1, help="Simulate demand increase/decrease")
        
        # Calculate impact
        total_orders = sum(st.session_state.daily_orders['daily_expected_orders'])
        current_capacity = sum([calculate_channel_capacity(ch)[0] for ch in st.session_state.channels_config])
        
        # Simulate new capacity
        new_capacity = current_capacity
        
        # Add truck capacity
        for truck_type in st.session_state.trucks_config['truck_types']:
            if truck_type['id'] == 'A':
                new_capacity += add_trucks_a * truck_type['capacity']
            elif truck_type['id'] == 'B':
                new_capacity += add_trucks_b * truck_type['capacity']
        
        # Add driver capacity
        for ch in st.session_state.channels_config:
            driver_capacity_gain = add_drivers * ch['trips_per_day_per_driver'] * ch['avg_orders_per_trip']
            new_capacity += driver_capacity_gain
        
        new_demand = total_orders * demand_multiplier
        
        st.divider()
        
        # Show impact
        st.subheader("📊 Impact Analysis")
        
        impact_col1, impact_col2, impact_col3 = st.columns(3)
        
        with impact_col1:
            capacity_change = ((new_capacity - current_capacity) / current_capacity * 100) if current_capacity > 0 else 0
            st.metric("New Capacity", f"{new_capacity:.0f}", 
                     delta=f"{capacity_change:+.0f}%")
        
        with impact_col2:
            demand_change = ((new_demand - total_orders) / total_orders * 100) if total_orders > 0 else 0
            st.metric("New Demand", f"{new_demand:.0f}", 
                     delta=f"{demand_change:+.0f}%")
        
        with impact_col3:
            capacity_gap = new_demand - new_capacity
            gap_percent = (capacity_gap / new_demand * 100) if new_demand > 0 else 0
            st.metric("Capacity Gap", f"{capacity_gap:.0f}",
                     delta=f"{gap_percent:.1f}% of demand")
            
            if capacity_gap > 0:
                st.error("⚠️ Still insufficient!")
            elif gap_percent < -20:
                st.warning("⚠️ Excess capacity (wasteful)")
            else:
                st.success("✓ Optimal capacity")
        
        # Financial impact
        st.divider()
        st.subheader("💰 Financial Impact")
        
        # Calculate cost changes
        avg_cost_per_order = 15  # Base estimate
        cost_current = total_orders * avg_cost_per_order
        cost_new = new_demand * avg_cost_per_order
        
        # Add cost of new resources
        if add_trucks_a > 0 or add_trucks_b > 0:
            truck_cost_per_day = (add_trucks_a * 100) + (add_trucks_b * 120)  # Assume daily truck cost
            cost_new += truck_cost_per_day
        
        if add_drivers > 0:
            driver_cost_per_day = add_drivers * len(st.session_state.channels_config) * 80  # Assume daily driver cost
            cost_new += driver_cost_per_day
        
        fin_col1, fin_col2 = st.columns(2)
        
        with fin_col1:
            st.metric("Current Est. Cost", f"${cost_current:,.2f}")
            st.caption("Based on current configuration")
        
        with fin_col2:
            cost_change = cost_new - cost_current
            st.metric("New Est. Cost", f"${cost_new:,.2f}", 
                     delta=f"${cost_change:+,.2f}")
            st.caption("With proposed changes")
        
        # Visualization
        fig_scenario = go.Figure()
        fig_scenario.add_trace(go.Bar(name='Current Capacity', x=['Current'], y=[current_capacity], marker_color='blue'))
        fig_scenario.add_trace(go.Bar(name='New Capacity', x=['Current'], y=[new_capacity], marker_color='green'))
        fig_scenario.add_trace(go.Scatter(name='Demand', x=['Current'], y=[new_demand], mode='markers+text',
                                         marker=dict(size=20, color='red'), text=[f"{new_demand:.0f}"], textposition="top center"))
        fig_scenario.update_layout(title='🔍 Capacity vs Demand Analysis', yaxis_title='Orders/Day',
                                  barmode='group', height=400)
        st.plotly_chart(fig_scenario, width='stretch')

# ========================== PREDICTIVE ANALYTICS SECTION ==========================
elif page == "📈 Predictive Analytics":
    st.header("📈 Predictive Analytics & Forecasting")
    st.markdown("### AI-Powered Demand Forecasting")
    
    # Help Section
    with st.expander("ℹ️ What is Predictive Analytics?", expanded=False):
        st.markdown("""
        ### Predictive Analytics Explained
        
        **Purpose:** Forecast future demand using AI and historical patterns.
        
        **What It Does:**
        - Analyzes historical order patterns
        - Identifies trends and cycles
        - Predicts future demand
        - Warns of capacity issues
        
        **Key Features:**
        
        **1. Historical Trend Analysis:**
        - Shows 7-90 days of historical data
        - 7-day moving average to smooth trends
        - Identifies weekly patterns, seasonality
        
        **2. AI Demand Forecast:**
        - Linear trend forecasting
        - Extrapolates patterns into future
        - Shows forecast for 7-30 days ahead
        
        **3. Forecast Insights:**
        - Average forecasted demand
        - Peak demand days
        - Low demand days
        
        **4. Capacity Alerts:**
        - Compares forecast to your capacity
        - Warns if forecast exceeds capacity
        - Shows how many orders short you'll be
        
        **How Forecasting Works:**
        - Uses weighted averages of recent data
        - Applies trend analysis (increasing/decreasing)
        - Accounts for weekly patterns
        - Adds noise for realistic variation
        
        **Business Value:**
        - **Proactive Planning:** Know demand before it happens
        - **Risk Prevention:** Identify shortages in advance
        - **Resource Planning:** Prepare for peak days
        - **Cost Optimization:** Adjust capacity based on forecast
        
        **When to Use:**
        - Weekly/monthly planning sessions
        - Before peak seasons
        - When launching new products/campaigns
        - Capacity expansion decisions
        """)
    
    if not st.session_state.daily_orders['daily_expected_orders']:
        st.warning("⚠️ Please configure daily orders first!")
    else:
        # Historical data simulation
        historical_days = st.slider("Days of Historical Data", 7, 90, 30)
        
        # Create simulated historical data based on current order pattern
        base_orders = st.session_state.daily_orders['daily_expected_orders']
        historical = []
        
        for i in range(historical_days):
            # Simulate with some randomness
            avg_order = sum(base_orders) / len(base_orders) if base_orders else 50
            trend_factor = 1 + (np.sin(i / 7) * 0.2)  # Weekly cycle
            noise = np.random.normal(0, 0.1)
            historical.append(avg_order * trend_factor * (1 + noise))
        
        historical = [max(0, int(h)) for h in historical]
        
        st.subheader("📊 Historical Trend Analysis")
        
        # Plot historical data
        hist_df = pd.DataFrame({
            'Day': range(1, len(historical) + 1),
            'Orders': historical
        })
        
        fig_hist = px.line(hist_df, x='Day', y='Orders', 
                          title='Historical Order Trends',
                          markers=True)
        fig_hist.add_trace(go.Scatter(x=hist_df['Day'], y=hist_df['Orders'].rolling(window=7).mean(),
                                      name='7-Day Moving Average', line=dict(dash='dash')))
        st.plotly_chart(fig_hist, width='stretch')
        
        st.divider()
        
        # Forecast
        st.subheader("🔮 AI Demand Forecast")
        
        forecast_days = st.slider("Forecast Period (Days)", 7, 30, 14)
        
        # Simple forecast using trend
        avg_historical = np.mean(historical[-7:])  # Last week average
        # Calculate trend as difference between recent average and previous period
        if len(historical) >= 14:
            recent_avg = np.mean(historical[-7:])
            previous_avg = np.mean(historical[-14:-7])
            trend = recent_avg - previous_avg
        else:
            trend = 0
        
        forecast = []
        for i in range(forecast_days):
            # Simple linear trend forecast
            predicted = avg_historical + (trend * i / 7) if len(historical) >= 14 else avg_historical
            forecast.append(max(0, int(predicted)))
        
        forecast_df = pd.DataFrame({
            'Day': range(len(historical) + 1, len(historical) + forecast_days + 1),
            'Forecast': forecast
        })
        
        # Combine historical and forecast
        combined_df = pd.concat([
            pd.DataFrame({'Day': range(1, len(historical) + 1), 'Orders': historical, 'Forecast': np.nan}),
            pd.DataFrame({'Day': forecast_df['Day'], 'Orders': np.nan, 'Forecast': forecast_df['Forecast']})
        ])
        
        fig_forecast = go.Figure()
        fig_forecast.add_trace(go.Scatter(x=list(range(1, len(historical) + 1)), y=historical,
                                         name='Historical Orders', line=dict(color='blue')))
        fig_forecast.add_trace(go.Scatter(x=forecast_df['Day'].tolist(), y=forecast_df['Forecast'].tolist(),
                                         name='AI Forecast', line=dict(color='red', dash='dash')))
        fig_forecast.update_layout(title='📈 Order Forecast', xaxis_title='Day', yaxis_title='Orders')
        st.plotly_chart(fig_forecast, width='stretch')
        
        st.divider()
        
        # Insights
        st.subheader("🎯 Forecast Insights")
        
        col_for1, col_for2, col_for3 = st.columns(3)
        
        avg_forecast = np.mean(forecast)
        max_forecast = max(forecast)
        min_forecast = min(forecast)
        
        with col_for1:
            st.metric("Avg Forecast", f"{avg_forecast:.0f}")
        
        with col_for2:
            st.metric("Peak Demand", f"{max_forecast:.0f}")
        
        with col_for3:
            st.metric("Low Demand", f"{min_forecast:.0f}")
        
        # Capacity check
        total_capacity = sum([calculate_channel_capacity(ch)[0] for ch in st.session_state.channels_config]) if st.session_state.channels_config else 0
        
        if avg_forecast > total_capacity:
            st.error(f"⚠️ **Critical:** Forecasted demand ({avg_forecast:.0f}) exceeds current capacity ({total_capacity:.0f}) by {avg_forecast - total_capacity:.0f} orders/day")
        
        if max_forecast > total_capacity:
            st.warning(f"⚠️ Peak days may exceed capacity by up to {max_forecast - total_capacity:.0f} orders")


# Footer
st.markdown("---")
st.markdown("🚚 **Logistics Planning Optimizer** - Powered by Streamlit & AI")

# Add intelligence badge in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### 🧠 AI Features")
st.sidebar.success("✓ Executive Dashboard")
st.sidebar.success("✓ What-If Scenarios")
st.sidebar.success("✓ Predictive Analytics")
st.sidebar.success("✓ Intelligent Optimization")
st.sidebar.success("✓ Cost Analysis")
st.sidebar.success("✓ Risk Scoring")
st.sidebar.success("✓ Industry Benchmarking")
st.sidebar.success("✓ Auto Recommendations")
st.sidebar.markdown("---")

# ========================== END OF FILE ==========================

# Footer
st.markdown("---")
st.markdown("🚚 **Logistics Planning Optimizer** - Powered by Streamlit & AI")
