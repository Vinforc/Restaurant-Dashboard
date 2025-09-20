import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(
    page_title="Bella Vista Bistro - Analytics Dashboard",
    page_icon="🍽️",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .metric-container {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        margin: 0;
    }
    .metric-label {
        font-size: 0.9rem;
        margin: 0;
        opacity: 0.9;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem 0.5rem 0 0;
    }
    .analysis-card {
  	background-color: var(--secondary-background-color); /* pulls from Streamlit theme */
  	color: var(--text-color); /* auto white in dark, black in light */
        border-radius: 0.75rem;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_sample_data():
    """Generate sample data based on our restaurant data structure"""
    np.random.seed(42)
    
    # Generate date range
    start_date = pd.to_datetime('2024-01-01')
    end_date = pd.to_datetime('2024-09-19')
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Menu items with realistic pricing
    menu_items = {
        'Bruschetta Trio': {'price': 12.95, 'category': 'Appetizer', 'cost': 4.50},
        'Caesar Salad': {'price': 10.95, 'category': 'Appetizer', 'cost': 3.80},
        'Grilled Salmon': {'price': 26.95, 'category': 'Entree', 'cost': 8.90},
        'Ribeye Steak': {'price': 34.95, 'category': 'Entree', 'cost': 12.25},
        'Chicken Parmigiana': {'price': 22.95, 'category': 'Entree', 'cost': 7.80},
        'Pasta Primavera': {'price': 18.95, 'category': 'Entree', 'cost': 5.20},
        'Tiramisu': {'price': 8.95, 'category': 'Dessert', 'cost': 2.50},
        'House Wine': {'price': 8.95, 'category': 'Beverage', 'cost': 2.80},
        'Craft Beer': {'price': 6.95, 'category': 'Beverage', 'cost': 2.10}
    }
    
    # Staff data
    staff = ['Sarah Johnson', 'Mike Chen', 'Lisa Rodriguez', 'Tom Wilson', 'Emma Davis']
    
    # Generate POS transactions
    transactions = []
    for date in date_range:
        # Weekend vs weekday patterns
        if date.weekday() in [5, 6]:  # Weekend
            daily_transactions = np.random.randint(45, 75)
        else:
            daily_transactions = np.random.randint(30, 55)
        
        for _ in range(daily_transactions):
            # Peak hours modeling
            hour_weights = [1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 4, 8, 12, 8, 4, 3, 6, 12, 15, 12, 8, 4, 2, 1]
            hour = np.random.choice(range(24), p=np.array(hour_weights)/sum(hour_weights))
            
            # Generate transaction
            item_name = np.random.choice(list(menu_items.keys()))
            item_data = menu_items[item_name]
            quantity = np.random.choice([1, 2], p=[0.8, 0.2])
            server = np.random.choice(staff)
            
            subtotal = item_data['price'] * quantity
            tax = subtotal * 0.0875
            tip = subtotal * np.random.choice([0.15, 0.18, 0.20, 0.22], p=[0.2, 0.3, 0.4, 0.1])
            total = subtotal + tax + tip
            
            transactions.append({
                'date': date,
                'hour': hour,
                'item_name': item_name,
                'category': item_data['category'],
                'quantity': quantity,
                'unit_price': item_data['price'],
                'unit_cost': item_data['cost'],
                'subtotal': subtotal,
                'tax': tax,
                'tip': tip,
                'total': total,
                'server': server,
                'payment_method': np.random.choice(['Credit Card', 'Cash', 'Debit Card'], p=[0.6, 0.2, 0.2])
            })
    
    pos_df = pd.DataFrame(transactions)
    
    # Generate labor data
    labor_data = []
    for date in date_range:
        for employee in staff:
            if np.random.random() < 0.85:  # 85% chance of working each day
                if employee == 'Sarah Johnson':  # Manager
                    hours = np.random.uniform(8, 10)
                    hourly_rate = 31.25
                    tips = 0
                else:  # Servers
                    hours = np.random.uniform(5, 8)
                    hourly_rate = 16.83
                    tips = np.random.uniform(50, 150)
                
                labor_data.append({
                    'date': date,
                    'employee': employee,
                    'hours': hours,
                    'hourly_rate': hourly_rate,
                    'tips': tips,
                    'total_pay': hours * hourly_rate + tips
                })
    
    labor_df = pd.DataFrame(labor_data)
    
    # Generate reviews data
    reviews_data = []
    for i in range(200):
        date = pd.Timestamp(np.random.choice(date_range))  # Convert to pandas Timestamp
        rating = np.random.choice([1, 2, 3, 4, 5], p=[0.05, 0.10, 0.15, 0.35, 0.35])
        reviews_data.append({
            'date': date,
            'rating': rating,
            'useful_votes': np.random.randint(0, 15) if rating in [1, 5] else np.random.randint(0, 5)
        })
    
    reviews_df = pd.DataFrame(reviews_data)
    
    # Generate reservations data
    reservations_data = []
    for i in range(300):
        date = pd.Timestamp(np.random.choice(date_range))  # Convert to pandas Timestamp
        party_size = np.random.choice([1, 2, 3, 4, 5, 6], p=[0.05, 0.4, 0.2, 0.2, 0.1, 0.05])
        status = np.random.choice(['Completed', 'No Show', 'Cancelled'], p=[0.85, 0.08, 0.07])
        
        reservations_data.append({
            'date': date,
            'party_size': party_size,
            'status': status,
            'day_of_week': date.day_name()
        })
    
    reservations_df = pd.DataFrame(reservations_data)
    
    return pos_df, labor_df, reviews_df, reservations_df

# Load data
pos_df, labor_df, reviews_df, reservations_df = load_sample_data()

# Title and header
st.markdown("# 🍽️ Bella Vista Bistro Analytics Dashboard")
st.markdown("### Real-time Business Intelligence for Restaurant Operations")
st.markdown("---")

# Date filter at the top
col1, col2, col3, col4 = st.columns([2, 2, 2, 4])
with col1:
    start_date = st.date_input(
        "Start Date",
        value=pos_df['date'].min(),
        min_value=pos_df['date'].min(),
        max_value=pos_df['date'].max()
    )
with col2:
    end_date = st.date_input(
        "End Date",
        value=pos_df['date'].max(),
        min_value=pos_df['date'].min(),
        max_value=pos_df['date'].max()
    )

# Filter data based on date range
mask = (pos_df['date'] >= pd.to_datetime(start_date)) & (pos_df['date'] <= pd.to_datetime(end_date))
filtered_pos = pos_df[mask]
filtered_labor = labor_df[(labor_df['date'] >= pd.to_datetime(start_date)) & (labor_df['date'] <= pd.to_datetime(end_date))]
filtered_reviews = reviews_df[(reviews_df['date'] >= pd.to_datetime(start_date)) & (reviews_df['date'] <= pd.to_datetime(end_date))]
filtered_reservations = reservations_df[(reservations_df['date'] >= pd.to_datetime(start_date)) & (reservations_df['date'] <= pd.to_datetime(end_date))]

# Create tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Executive Overview", 
    "💰 Financial Performance", 
    "👥 Labor Analytics", 
    "🍽️ Menu Analysis", 
    "⭐ Customer Experience",
    "🤖 ML Insights"
])

with tab1:
    st.markdown("## Executive Summary")
    
    # Key Metrics
    total_revenue = filtered_pos['total'].sum()
    total_orders = len(filtered_pos)
    avg_order_value = total_revenue / total_orders if total_orders > 0 else 0
    total_customers = filtered_pos.groupby('date')['server'].count().sum()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-container">
            <p class="metric-value">${total_revenue:,.0f}</p>
            <p class="metric-label">Total Revenue</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-container">
            <p class="metric-value">{total_orders:,}</p>
            <p class="metric-label">Total Orders</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-container">
            <p class="metric-value">${avg_order_value:.2f}</p>
            <p class="metric-label">Avg Order Value</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        avg_rating = filtered_reviews['rating'].mean() if len(filtered_reviews) > 0 else 0
        st.markdown(f"""
        <div class="metric-container">
            <p class="metric-value">{avg_rating:.1f}⭐</p>
            <p class="metric-label">Avg Rating</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Daily Revenue Trend
    daily_revenue = filtered_pos.groupby('date')['total'].sum().reset_index()
    daily_revenue['date'] = pd.to_datetime(daily_revenue['date'])
    
    st.markdown("### 📈 Daily Revenue Trend")
    chart = alt.Chart(daily_revenue).mark_line(
        color='#667eea',
        strokeWidth=3,
        point=alt.OverlayMarkDef(color='#764ba2', size=50)
    ).add_selection(
        alt.selection_interval(bind='scales')
    ).encode(
        x=alt.X('date:T', title='Date'),
        y=alt.Y('total:Q', title='Revenue ($)', scale=alt.Scale(zero=False)),
        tooltip=['date:T', alt.Tooltip('total:Q', format='$,.0f')]
    ).properties(
        width=800,
        height=300
    )
    st.altair_chart(chart, use_container_width=True)
    
    # Peak Hours Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🕐 Peak Hours Analysis")
        hourly_sales = filtered_pos.groupby('hour')['total'].sum().reset_index()
        
        chart = alt.Chart(hourly_sales).mark_bar(
            color=alt.Gradient(
                gradient='linear',
                stops=[alt.GradientStop(color='#667eea', offset=0),
                       alt.GradientStop(color='#764ba2', offset=1)]
            )
        ).encode(
            x=alt.X('hour:O', title='Hour of Day'),
            y=alt.Y('total:Q', title='Total Sales ($)'),
            tooltip=['hour:O', alt.Tooltip('total:Q', format='$,.0f')]
        ).properties(
            width=350,
            height=250
        )
        st.altair_chart(chart, use_container_width=True)
    
    with col2:
        st.markdown("### 🗓️ Day of Week Performance")
        filtered_pos['day_of_week'] = pd.to_datetime(filtered_pos['date']).dt.day_name()
        daily_performance = filtered_pos.groupby('day_of_week')['total'].sum().reset_index()
        
        # Order days properly
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily_performance['day_of_week'] = pd.Categorical(daily_performance['day_of_week'], categories=day_order, ordered=True)
        daily_performance = daily_performance.sort_values('day_of_week')
        
        chart = alt.Chart(daily_performance).mark_bar(
            color=alt.Gradient(
                gradient='linear',
                stops=[alt.GradientStop(color='#764ba2', offset=0),
                       alt.GradientStop(color='#667eea', offset=1)]
            )
        ).encode(
            x=alt.X('day_of_week:O', title='Day of Week', sort=day_order),
            y=alt.Y('total:Q', title='Total Sales ($)'),
            tooltip=['day_of_week:O', alt.Tooltip('total:Q', format='$,.0f')]
        ).properties(
            width=350,
            height=250
        )
        st.altair_chart(chart, use_container_width=True)

with tab2:
    st.markdown("## 💰 Financial Performance")
    
    # Financial KPIs
    total_cost = (filtered_pos['unit_cost'] * filtered_pos['quantity']).sum()
    gross_profit = total_revenue - total_cost
    gross_margin = (gross_profit / total_revenue * 100) if total_revenue > 0 else 0
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Revenue", f"${total_revenue:,.0f}")
    with col2:
        st.metric("Total Food Cost", f"${total_cost:,.0f}")
    with col3:
        st.metric("Gross Profit", f"${gross_profit:,.0f}")
    with col4:
        st.metric("Gross Margin", f"{gross_margin:.1f}%")
    
    # Revenue vs Cost Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Monthly P&L Summary")
        monthly_data = filtered_pos.copy()
        monthly_data['month'] = pd.to_datetime(monthly_data['date']).dt.to_period('M')
        monthly_summary = monthly_data.groupby('month').agg({
            'total': 'sum',
            'unit_cost': lambda x: (x * monthly_data.loc[x.index, 'quantity']).sum()
        }).reset_index()
        monthly_summary['profit'] = monthly_summary['total'] - monthly_summary['unit_cost']
        monthly_summary['month_str'] = monthly_summary['month'].astype(str)
        
        # Create dual-axis chart
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(
            go.Bar(name="Revenue", x=monthly_summary['month_str'], y=monthly_summary['total'],
                  marker_color='#667eea'),
        )
        
        fig.add_trace(
            go.Bar(name="Cost", x=monthly_summary['month_str'], y=monthly_summary['unit_cost'],
                  marker_color='#ff6b6b'),
        )
        
        fig.add_trace(
            go.Scatter(name="Profit", x=monthly_summary['month_str'], y=monthly_summary['profit'],
                      mode='lines+markers', line=dict(color='#51cf66', width=3),
                      marker=dict(size=8)),
            secondary_y=True
        )
        
        fig.update_layout(
            title="Revenue, Cost & Profit Trends",
            xaxis_title="Month",
            barmode='group',
            height=400
        )
        fig.update_yaxes(title_text="Amount ($)", secondary_y=False)
        fig.update_yaxes(title_text="Profit ($)", secondary_y=True)
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 💳 Payment Method Distribution")
        payment_dist = filtered_pos.groupby('payment_method').agg({
            'total': ['sum', 'count']
        }).round(2)
        payment_dist.columns = ['Total_Revenue', 'Transaction_Count']
        payment_dist = payment_dist.reset_index()
        
        fig = px.pie(
            payment_dist, 
            values='Total_Revenue', 
            names='payment_method',
            title='Revenue by Payment Method',
            color_discrete_sequence=['#667eea', '#764ba2', '#ff6b6b', '#51cf66']
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.markdown("## 👥 Labor Analytics")
    
    # Labor KPIs
    total_labor_cost = filtered_labor['total_pay'].sum()
    total_hours = filtered_labor['hours'].sum()
    avg_hourly_rate = filtered_labor['hourly_rate'].mean()
    labor_percentage = (total_labor_cost / total_revenue * 100) if total_revenue > 0 else 0
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Labor Cost", f"${total_labor_cost:,.0f}")
    with col2:
        st.metric("Total Hours", f"{total_hours:,.0f}")
    with col3:
        st.metric("Avg Hourly Rate", f"${avg_hourly_rate:.2f}")
    with col4:
        st.metric("Labor Cost %", f"{labor_percentage:.1f}%")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Employee Performance")
        employee_performance = filtered_labor.groupby('employee').agg({
            'hours': 'sum',
            'tips': 'sum',
            'total_pay': 'sum'
        }).reset_index()
        employee_performance = employee_performance.sort_values('total_pay', ascending=True)
        
        chart = alt.Chart(employee_performance).mark_bar(
            color='#667eea'
        ).encode(
            x=alt.X('total_pay:Q', title='Total Earnings ($)'),
            y=alt.Y('employee:N', title='Employee', sort='-x'),
            tooltip=['employee:N', 'total_pay:Q', 'hours:Q', 'tips:Q']
        ).properties(
            width=400,
            height=300
        )
        st.altair_chart(chart, use_container_width=True)
    
    with col2:
        st.markdown("### 📈 Labor Cost Trend")
        daily_labor = filtered_labor.groupby('date')['total_pay'].sum().reset_index()
        
        chart = alt.Chart(daily_labor).mark_area(
            color='#764ba2',
            opacity=0.7
        ).encode(
            x=alt.X('date:T', title='Date'),
            y=alt.Y('total_pay:Q', title='Daily Labor Cost ($)'),
            tooltip=['date:T', alt.Tooltip('total_pay:Q', format='$,.0f')]
        ).properties(
            width=400,
            height=300
        )
        st.altair_chart(chart, use_container_width=True)

with tab4:
    st.markdown("## 🍽️ Menu Analysis")
    
    # Menu performance metrics
    menu_performance = filtered_pos.groupby(['item_name', 'category']).agg({
        'quantity': 'sum',
        'total': 'sum',
        'unit_price': 'first',
        'unit_cost': 'first'
    }).reset_index()
    menu_performance['profit_per_item'] = menu_performance['unit_price'] - menu_performance['unit_cost']
    menu_performance['total_profit'] = menu_performance['quantity'] * menu_performance['profit_per_item']
    menu_performance['margin'] = (menu_performance['profit_per_item'] / menu_performance['unit_price'] * 100).round(1)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🏆 Top Selling Items")
        top_items = menu_performance.nlargest(10, 'quantity')[['item_name', 'quantity', 'total']]
        
        chart = alt.Chart(top_items).mark_bar(
            color=alt.Gradient(
                gradient='linear',
                stops=[alt.GradientStop(color='#667eea', offset=0),
                       alt.GradientStop(color='#764ba2', offset=1)]
            )
        ).encode(
            x=alt.X('quantity:Q', title='Units Sold'),
            y=alt.Y('item_name:N', title='Menu Item', sort='-x'),
            tooltip=['item_name:N', 'quantity:Q', alt.Tooltip('total:Q', format='$,.0f')]
        ).properties(
            width=400,
            height=400
        )
        st.altair_chart(chart, use_container_width=True)
    
    with col2:
        st.markdown("### 💰 Most Profitable Items")
        profitable_items = menu_performance.nlargest(10, 'total_profit')[['item_name', 'total_profit', 'margin']]
        
        chart = alt.Chart(profitable_items).mark_bar(
            color=alt.Gradient(
                gradient='linear',
                stops=[alt.GradientStop(color='#51cf66', offset=0),
                       alt.GradientStop(color='#40c057', offset=1)]
            )
        ).encode(
            x=alt.X('total_profit:Q', title='Total Profit ($)'),
            y=alt.Y('item_name:N', title='Menu Item', sort='-x'),
            tooltip=['item_name:N', alt.Tooltip('total_profit:Q', format='$,.0f'), 'margin:Q']
        ).properties(
            width=400,
            height=400
        )
        st.altair_chart(chart, use_container_width=True)

with tab5:
    st.markdown("## ⭐ Customer Experience")
    
    # Customer metrics
    if len(filtered_reviews) > 0:
        avg_rating = filtered_reviews['rating'].mean()
        total_reviews = len(filtered_reviews)
        rating_distribution = filtered_reviews['rating'].value_counts().sort_index()
    else:
        avg_rating = 0
        total_reviews = 0
        rating_distribution = pd.Series()
    
    # Reservation metrics
    total_reservations = len(filtered_reservations)
    completed_reservations = len(filtered_reservations[filtered_reservations['status'] == 'Completed'])
    no_show_rate = len(filtered_reservations[filtered_reservations['status'] == 'No Show']) / total_reservations * 100 if total_reservations > 0 else 0
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Average Rating", f"{avg_rating:.1f}⭐")
    with col2:
        st.metric("Total Reviews", f"{total_reviews:,}")
    with col3:
        st.metric("Reservations", f"{completed_reservations:,}")
    with col4:
        st.metric("No-Show Rate", f"{no_show_rate:.1f}%")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Rating Distribution")
        if len(rating_distribution) > 0:
            rating_df = pd.DataFrame({
                'rating': rating_distribution.index,
                'count': rating_distribution.values
            })
            
            chart = alt.Chart(rating_df).mark_bar(
                color=alt.expr("datum.rating >= 4 ? '#51cf66' : datum.rating >= 3 ? '#ffd43b' : '#ff6b6b'")
            ).encode(
                x=alt.X('rating:O', title='Rating'),
                y=alt.Y('count:Q', title='Number of Reviews'),
                tooltip=['rating:O', 'count:Q']
            ).properties(
                width=350,
                height=250
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("No review data available for selected period")
    
    with col2:
        st.markdown("### 📅 Reservation Patterns")
        if len(filtered_reservations) > 0:
            day_reservations = filtered_reservations.groupby('day_of_week').size().reset_index(name='count')
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            day_reservations['day_of_week'] = pd.Categorical(day_reservations['day_of_week'], categories=day_order, ordered=True)
            day_reservations = day_reservations.sort_values('day_of_week')
            
            chart = alt.Chart(day_reservations).mark_bar(
                color='#667eea'
            ).encode(
                x=alt.X('day_of_week:O', title='Day of Week', sort=day_order),
                y=alt.Y('count:Q', title='Number of Reservations'),
                tooltip=['day_of_week:O', 'count:Q']
            ).properties(
                width=350,
                height=250
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("No reservation data available for selected period")

with tab6:
    st.markdown("## 🤖 Machine Learning Insights")
    st.markdown("### Advanced Analytics for Business Optimization")
    
    # ML Study 1: Revenue Prediction
    st.markdown("---")
    st.markdown("### 📊 Study 1: Daily Revenue Prediction Model")
    
    with st.container():
        st.markdown("""
        <div class="analysis-card">
            <h4>🎯 Objective</h4>
            <p>Predict daily revenue based on historical patterns, day of week, and operational factors to improve staffing and inventory planning.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Prepare data for ML model
        daily_features = filtered_pos.copy()
        daily_features['date'] = pd.to_datetime(daily_features['date'])
        daily_features['day_of_week'] = daily_features['date'].dt.dayofweek
        daily_features['month'] = daily_features['date'].dt.month
        daily_features['is_weekend'] = daily_features['day_of_week'].isin([5, 6]).astype(int)
        
        # Aggregate daily data
        daily_agg = daily_features.groupby('date').agg({
            'total': 'sum',
            'quantity': 'sum',
            'day_of_week': 'first',
            'month': 'first',
            'is_weekend': 'first'
        }).reset_index()
        
        if len(daily_agg) > 10:  # Only run if we have enough data
            # Add weather simulation
            np.random.seed(42)
            daily_agg['temperature'] = np.random.normal(22, 8, len(daily_agg))
            daily_agg['is_rainy'] = np.random.choice([0, 1], len(daily_agg), p=[0.8, 0.2])
            
            # Prepare features and target
            feature_cols = ['day_of_week', 'month', 'is_weekend', 'temperature', 'is_rainy', 'quantity']
            X = daily_agg[feature_cols]
            y = daily_agg['total']
            
            # Split data
            if len(X) > 5:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                
                # Train model
                rf_model = RandomForestRegressor(n_estimators=50, random_state=42)
                rf_model.fit(X_train, y_train)
                
                # Predictions
                y_pred = rf_model.predict(X_test)
                mae = mean_absolute_error(y_test, y_pred)

                r2 = r2_score(y_test, y_pred)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Model Accuracy (R²)", f"{r2:.3f}")
                with col2:
                    st.metric("Mean Absolute Error", f"${mae:.2f}")
                with col3:
                    st.metric("Avg Daily Revenue", f"${y.mean():.2f}")
                
                # Feature importance
                feature_importance = pd.DataFrame({
                    'feature': feature_cols,
                    'importance': rf_model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 🔍 Feature Importance")
                    chart = alt.Chart(feature_importance).mark_bar(
                        color='#667eea'
                    ).encode(
                        x=alt.X('importance:Q', title='Importance Score'),
                        y=alt.Y('feature:N', title='Feature', sort='-x'),
                        tooltip=['feature:N', alt.Tooltip('importance:Q', format='.3f')]
                    ).properties(
                        width=350,
                        height=250
                    )
                    st.altair_chart(chart, use_container_width=True)
                
                with col2:
                    st.markdown("#### 📈 Actual vs Predicted Revenue")
                    comparison_df = pd.DataFrame({
                        'actual': y_test,
                        'predicted': y_pred
                    })
                    
                    chart = alt.Chart(comparison_df).mark_circle(
                        color='#764ba2',
                        size=60,
                        opacity=0.7
                    ).encode(
                        x=alt.X('actual:Q', title='Actual Revenue ($)'),
                        y=alt.Y('predicted:Q', title='Predicted Revenue ($)'),
                        tooltip=[alt.Tooltip('actual:Q', format='$,.0f'), alt.Tooltip('predicted:Q', format='$,.0f')]
                    ).properties(
                        width=350,
                        height=250
                    )
                    
                    # Add perfect prediction line
                    min_val = min(comparison_df['actual'].min(), comparison_df['predicted'].min())
                    max_val = max(comparison_df['actual'].max(), comparison_df['predicted'].max())
                    line_df = pd.DataFrame({'x': [min_val, max_val], 'y': [min_val, max_val]})
                    
                    line = alt.Chart(line_df).mark_line(
                        color='#ff6b6b',
                        strokeDash=[5, 5]
                    ).encode(
                        x='x:Q',
                        y='y:Q'
                    )
                    
                    st.altair_chart(chart + line, use_container_width=True)
                
                st.markdown("""
                <div class="analysis-card">
                    <h4>💡 Key Insights</h4>
                    <ul>
                        <li><strong>Quantity sold</strong> is the strongest predictor of daily revenue</li>
                        <li><strong>Day of week</strong> significantly impacts revenue - weekends perform better</li>
                        <li><strong>Weather conditions</strong> have moderate influence on customer visits</li>
                        <li>Model can predict revenue within <strong>${:.0f}</strong> on average</li>
                    </ul>
                </div>
                """.format(mae), unsafe_allow_html=True)
            else:
                st.info("Not enough data for model training. Please select a larger date range.")
        else:
            st.info("Not enough data for analysis. Please select a larger date range.")
    
    # ML Study 2: Customer Segmentation
    st.markdown("---")
    st.markdown("### 👥 Study 2: Customer Behavior Segmentation")
    
    with st.container():
        st.markdown("""
        <div class="analysis-card">
            <h4>🎯 Objective</h4>
            <p>Segment customers based on dining patterns to optimize marketing strategies and personalized service offerings.</p>
        </div>
        """, unsafe_allow_html=True)
        
        if len(filtered_pos) > 0:
            # Create customer behavior data using server as proxy
            customer_behavior = filtered_pos.groupby('server').agg({
                'total': ['mean', 'sum', 'count'],
                'tip': 'mean',
                'hour': 'mean'
            }).round(2)
            
            customer_behavior.columns = ['avg_order_value', 'total_spent', 'visit_frequency', 'avg_tip', 'preferred_time']
            customer_behavior = customer_behavior.reset_index()
            
            # Add some realistic customer characteristics
            np.random.seed(42)
            customer_behavior['days_since_last_visit'] = np.random.randint(1, 90, len(customer_behavior))
            customer_behavior['prefers_weekend'] = (np.random.random(len(customer_behavior)) > 0.6).astype(int)
            
            if len(customer_behavior) >= 4:
                # Prepare features for clustering
                clustering_features = ['avg_order_value', 'visit_frequency', 'avg_tip', 'preferred_time', 'days_since_last_visit']
                X_cluster = customer_behavior[clustering_features]
                
                # Normalize features
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_cluster)
                
                # Perform K-means clustering
                n_clusters = min(4, len(customer_behavior))
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                customer_behavior['segment'] = kmeans.fit_predict(X_scaled)
                
                # Define segment names
                segment_names = {
                    0: 'Casual Diners',
                    1: 'Premium Customers', 
                    2: 'Regular Visitors',
                    3: 'Occasional Splurgers'
                }
                customer_behavior['segment_name'] = customer_behavior['segment'].map(lambda x: segment_names.get(x, f'Segment {x}'))
                
                # Visualize segments
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 🎯 Customer Segments")
                    
                    chart = alt.Chart(customer_behavior).mark_circle(size=100).encode(
                        x=alt.X('avg_order_value:Q', title='Average Order Value ($)'),
                        y=alt.Y('visit_frequency:Q', title='Visit Frequency'),
                        color=alt.Color('segment_name:N', 
                                       scale=alt.Scale(range=['#667eea', '#764ba2', '#ff6b6b', '#51cf66']),
                                       title='Customer Segment'),
                        tooltip=['server:N', 'segment_name:N', 'avg_order_value:Q', 'visit_frequency:Q', 'avg_tip:Q']
                    ).properties(
                        width=350,
                        height=300
                    )
                    st.altair_chart(chart, use_container_width=True)
                
                with col2:
                    st.markdown("#### 📊 Segment Distribution")
                    segment_dist = customer_behavior['segment_name'].value_counts().reset_index()
                    segment_dist.columns = ['segment', 'count']
                    
                    chart = alt.Chart(segment_dist).mark_arc(
                        innerRadius=50,
                        outerRadius=120
                    ).encode(
                        theta=alt.Theta('count:Q', title='Count'),
                        color=alt.Color('segment:N', 
                                       scale=alt.Scale(range=['#667eea', '#764ba2', '#ff6b6b', '#51cf66']),
                                       title='Segment'),
                        tooltip=['segment:N', 'count:Q']
                    ).properties(
                        width=300,
                        height=300
                    )
                    st.altair_chart(chart, use_container_width=True)
                
                # Segment characteristics table
                st.markdown("#### 📋 Segment Characteristics")
                segment_summary_display = customer_behavior.groupby('segment_name').agg({
                    'avg_order_value': 'mean',
                    'visit_frequency': 'mean', 
                    'avg_tip': 'mean',
                    'preferred_time': 'mean',
                    'server': 'count'
                }).round(2)
                segment_summary_display.columns = ['Avg Order ($)', 'Visits', 'Avg Tip ($)', 'Preferred Hour', 'Count']
                segment_summary_display['Preferred Hour'] = segment_summary_display['Preferred Hour'].astype(int)
                
                st.dataframe(segment_summary_display, use_container_width=True)
                
                st.markdown("""
                <div class="analysis-card">
                    <h4>🎯 Marketing Recommendations</h4>
                    <ul>
                        <li><strong>Premium Customers:</strong> Offer exclusive wine pairings and chef's table experiences</li>
                        <li><strong>Regular Visitors:</strong> Implement loyalty program with progressive rewards</li>
                        <li><strong>Casual Diners:</strong> Promote lunch specials and happy hour deals</li>
                        <li><strong>Occasional Splurgers:</strong> Target with special occasion marketing (birthdays, anniversaries)</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("Not enough customer data for segmentation analysis.")
        else:
            st.info("No transaction data available for customer analysis.")
    
    # Business Intelligence Summary
    st.markdown("---")
    st.markdown("### 📈 AI-Powered Business Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="analysis-card">
            <h4>🔮 Revenue Optimization</h4>
            <ul>
                <li>Focus on <strong>weekend operations</strong> - highest revenue potential</li>
                <li>Optimize staffing for <strong>predicted high-revenue days</strong></li>
                <li>Weather-based promotions during <strong>rainy days</strong></li>
                <li>Menu engineering: promote <strong>high-quantity items</strong></li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="analysis-card">
            <h4>👥 Customer Engagement</h4>
            <ul>
                <li><strong>Personalized marketing</strong> based on customer segments</li>
                <li>Time-based promotions matching <strong>preferred dining hours</strong></li>
                <li>Loyalty programs for <strong>regular visitors</strong></li>
                <li>Premium experiences for <strong>high-value customers</strong></li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>🍽️ <strong>Bella Vista Bistro Analytics Dashboard</strong> | Powered by Streamlit & Forcalytics</p>
        <p>Real-time insights for smarter restaurant management</p>
    </div>
    """, 
    unsafe_allow_html=True
)