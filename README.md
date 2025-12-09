# 🚚 Logistics Planning Optimizer

A comprehensive Streamlit application for logistics planning and optimization.

## Quick Start

### Option 1: Double-Click to Run (Easiest)
Simply **double-click** on `run_app.bat` - the app will start automatically and open in your browser!

### Option 2: Command Line
Open PowerShell or Command Prompt in this folder and run:
```bash
streamlit run Logistic.py
```

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

If this is your first time, you may need to install Python packages. The app will show you what to install.

## Features

### 1. 📋 Daily Orders
- Configure planning period (X days)
- Enter daily expected orders per day
- Visualize order distribution
- Track total and average orders

### 2. 📅 Daily Slots
- Configure number of slots per day
- Set weight for each slot
- Weight distribution validation
- Visual slot weight distribution

### 3. 🚛 Trucks Configuration
- Configure number of available trucks in network
- Define truck types with max orders per day
- Truck A (4 orders) - configurable
- Truck B (6 orders) - configurable
- Fleet capacity calculations

### 4. 🔗 Channel Configuration
- Multiple channels support
- For each channel:
  - **Channel ID**: Unique identifier
  - **Cost per order**: Variable cost
  - **Fixed cost per trip**: Fixed cost per trip
  - **Drivers available**: Number of drivers
  - **Available trucks per type**: Type A and Type B
  - **Trips per day per driver**: Daily trip capacity
  - **Average orders per trip**: Order capacity per trip
  - **OTP Metrics**: 24 hours / 7 days / 30 days performance

### 5. ⚙️ Optimization Parameters
- **OTP Time Window Weights**: Balance 24h/7d/30d metrics
- **Target Scenarios**: Recommend plans for X% and Y% OTP
- **Cost vs OTP Weight**: 
  - 0 = Maximize OTP (minimum cost)
  - 1 = Minimize Cost (minimum OTP)

### 6. 📊 AI Optimization & Insights
- Generate optimization scenarios
- Configuration summaries
- Performance visualizations
- OTP performance by channel
- Intelligent recommendations
- Cost analysis
- Risk scoring

### 7. 🧪 What-If Scenarios
- Test different capacity configurations
- See financial impact of changes
- Simulate demand changes
- Compare scenarios side-by-side

### 8. 📈 Predictive Analytics
- Historical trend analysis
- AI-powered demand forecasting
- Forecast insights
- Capacity alerts

### 9. 🏆 Executive Dashboard
- One-page strategic overview
- Key Performance Indicators
- Industry benchmarking
- Risk scoring
- Strategic recommendations
- Scenario comparison

## 💡 Help & Guidance

Each section has an **ℹ️ "What is this section?"** expandable box that explains:
- What the section does
- Key concepts and terminology
- How to use it
- Business value and impact

**Just click the info button at the top of each section!**

## Navigation

Use the sidebar to navigate between sections:
- **🏆 Executive Dashboard** - Strategic overview
- **📋 Daily Orders** - Configure expected orders
- **📅 Daily Slots** - Set time slot configurations
- **🚛 Trucks Configuration** - Define your fleet
- **🔗 Channel Configuration** - Set up delivery channels
- **⚙️ Optimization Parameters** - Configure AI optimization
- **📊 AI Optimization & Insights** - Run optimization and view results
- **🧪 What-If Scenarios** - Test different strategies
- **📈 Predictive Analytics** - Forecast future demand

## Export Features

- **Export to CSV**: Download optimization plans
- **Save Configuration**: Export your settings
- **Load Configuration**: Import saved settings

## Smart Features

✅ **Intelligent Optimization** - AI-driven order allocation  
✅ **Cost Analysis** - Real-time cost calculations  
✅ **OTP Prediction** - Performance forecasting  
✅ **Capacity Planning** - Identify bottlenecks  
✅ **Risk Assessment** - Automated risk detection  
✅ **Industry Benchmarking** - Compare to market standards  
✅ **Auto Recommendations** - AI-powered suggestions  
✅ **What-If Analysis** - Test scenarios before committing  
✅ **Predictive Analytics** - Forecast future demand  

## Requirements

- Python 3.8+
- Streamlit
- pandas
- numpy
- plotly

## Support

For issues or questions:
1. Check the help sections (ℹ️) in each module
2. Review the explanations for each feature
3. Use What-If Scenarios to test changes

---

**Powered by Streamlit & AI** 🚚