def get_custom_css() -> str:
    """
    Returns premium FinTech + Cybersecurity + AI aesthetic CSS styling.
    Designed for dark-mode enterprise financial risk surveillance dashboards.
    """
    return """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600;700&display=swap');
        
        /* Base page adjustments */
        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }
        
        .main .block-container {
            padding-top: 1.5rem;
            padding-bottom: 3.5rem;
            max-width: 1280px;
        }
        
        /* Monospace accents for numbers & metrics */
        .metric-value, .mono-text {
            font-family: 'JetBrains Mono', monospace;
        }

        /* Sidebar Styling */
        section[data-testid="stSidebar"] {
            background-color: #0d121f !important;
            border-right: 1px solid rgba(255, 255, 255, 0.07);
        }
        
        .sidebar-brand-card {
            background: linear-gradient(145deg, #111a2e 0%, #0d1424 100%);
            border: 1px solid rgba(0, 242, 254, 0.25);
            border-radius: 12px;
            padding: 16px 14px;
            margin-bottom: 20px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
            text-align: center;
        }
        
        .brand-icon {
            font-size: 1.8rem;
            margin-bottom: 4px;
            display: inline-block;
            filter: drop-shadow(0 0 10px rgba(0, 242, 254, 0.4));
        }
        
        .brand-name {
            font-size: 1.3rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            background: linear-gradient(90deg, #00f2fe 0%, #4facfe 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 0;
            line-height: 1.2;
        }
        
        .brand-title {
            font-size: 0.75rem;
            font-weight: 600;
            color: #94a3b8;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin-top: 4px;
        }
        
        .brand-pill {
            display: inline-block;
            background: rgba(0, 242, 254, 0.12);
            color: #38bdf8;
            border: 1px solid rgba(56, 189, 248, 0.3);
            font-size: 0.65rem;
            font-weight: 700;
            padding: 2px 8px;
            border-radius: 20px;
            margin-top: 8px;
            letter-spacing: 0.05em;
        }
        
        /* System status container */
        .system-status-box {
            background: #090e17;
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 10px;
            padding: 12px;
            margin-top: 15px;
        }
        
        .status-item {
            display: flex;
            align-items: center;
            font-size: 0.76rem;
            color: #94a3b8;
            margin-bottom: 6px;
        }
        .status-item:last-child {
            margin-bottom: 0;
        }
        
        .status-dot {
            width: 7px;
            height: 7px;
            border-radius: 50%;
            margin-right: 8px;
            display: inline-block;
        }
        .dot-green {
            background-color: #10b981;
            box-shadow: 0 0 8px rgba(16, 185, 129, 0.8);
        }
        .dot-cyan {
            background-color: #00f2fe;
            box-shadow: 0 0 8px rgba(0, 242, 254, 0.8);
        }
        
        /* Top Header Banner */
        .academic-banner {
            background: linear-gradient(135deg, #0e172a 0%, #132238 60%, #0a1120 100%);
            border: 1px solid rgba(0, 242, 254, 0.2);
            border-left: 4px solid #00f2fe;
            padding: 22px 26px;
            border-radius: 14px;
            margin-bottom: 24px;
            box-shadow: 0 8px 30px rgba(0, 0, 0, 0.4);
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 15px;
        }
        
        .academic-title {
            font-size: 1.45rem;
            font-weight: 800;
            letter-spacing: -0.01em;
            color: #ffffff;
            margin: 0 0 4px 0;
            line-height: 1.25;
        }
        
        .academic-subtitle {
            font-size: 0.85rem;
            color: #94a3b8;
            margin: 0;
            font-weight: 500;
        }
        
        .academic-badge {
            background: rgba(0, 242, 254, 0.08);
            border: 1px solid rgba(0, 242, 254, 0.25);
            color: #38bdf8;
            padding: 6px 14px;
            border-radius: 8px;
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.06em;
            text-transform: uppercase;
        }

        /* Metric / KPI Card Styling */
        .metric-card {
            background: linear-gradient(160deg, #131d31 0%, #0d1424 100%);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 12px;
            padding: 18px 20px;
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.25);
            transition: all 0.25s ease;
            position: relative;
            overflow: hidden;
            height: 100%;
        }
        
        .metric-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: linear-gradient(90deg, #00f2fe 0%, #4facfe 100%);
            opacity: 0.85;
        }
        
        .metric-card:hover {
            transform: translateY(-3px);
            border-color: rgba(0, 242, 254, 0.35);
            box-shadow: 0 10px 25px rgba(0, 242, 254, 0.12);
        }
        
        .metric-title {
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #94a3b8;
            font-weight: 700;
            margin-bottom: 6px;
            display: flex;
            align-items: center;
            gap: 6px;
        }
        
        .metric-value {
            font-size: 1.85rem;
            font-weight: 800;
            color: #f8fafc;
            line-height: 1.15;
            letter-spacing: -0.02em;
        }
        
        .metric-sub {
            font-size: 0.75rem;
            color: #64748b;
            margin-top: 6px;
            font-weight: 500;
        }

        /* Live Monitoring Summary Panel */
        .monitor-panel {
            background: #0f172a;
            border: 1px solid rgba(255, 255, 255, 0.07);
            border-radius: 10px;
            padding: 12px 18px;
            margin-bottom: 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 0.82rem;
            color: #cbd5e1;
        }

        /* Risk Badges */
        .badge-low {
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(6, 95, 70, 0.25) 100%);
            color: #34d399;
            padding: 8px 18px;
            border-radius: 8px;
            font-weight: 800;
            display: inline-block;
            font-size: 1.1rem;
            border: 1px solid rgba(52, 211, 153, 0.4);
            letter-spacing: 0.04em;
            box-shadow: 0 0 15px rgba(16, 185, 129, 0.2);
        }
        
        .badge-medium {
            background: linear-gradient(135deg, rgba(245, 158, 11, 0.15) 0%, rgba(146, 64, 14, 0.25) 100%);
            color: #fbbf24;
            padding: 8px 18px;
            border-radius: 8px;
            font-weight: 800;
            display: inline-block;
            font-size: 1.1rem;
            border: 1px solid rgba(251, 191, 36, 0.4);
            letter-spacing: 0.04em;
            box-shadow: 0 0 15px rgba(245, 158, 11, 0.2);
        }
        
        .badge-high {
            background: linear-gradient(135deg, rgba(239, 68, 68, 0.18) 0%, rgba(153, 27, 27, 0.3) 100%);
            color: #f87171;
            padding: 8px 18px;
            border-radius: 8px;
            font-weight: 800;
            display: inline-block;
            font-size: 1.1rem;
            border: 1px solid rgba(248, 113, 113, 0.5);
            letter-spacing: 0.04em;
            box-shadow: 0 0 18px rgba(239, 68, 68, 0.3);
        }

        /* Risk Result Container */
        .risk-assessment-container {
            background: #0f182c;
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 14px;
            padding: 24px;
            margin-bottom: 24px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
        }

        /* Explainability Indicator Boxes */
        .indicator-box {
            background: #0d1424;
            border: 1px solid rgba(255, 255, 255, 0.06);
            border-left: 4px solid #38bdf8;
            padding: 12px 18px;
            border-radius: 0 10px 10px 0;
            margin-bottom: 10px;
            transition: transform 0.15s ease;
        }
        .indicator-box:hover {
            transform: translateX(4px);
        }
        
        .indicator-critical {
            border-left-color: #ef4444;
            background: rgba(239, 68, 68, 0.06);
        }
        .indicator-high {
            border-left-color: #f97316;
            background: rgba(249, 115, 22, 0.06);
        }
        .indicator-medium {
            border-left-color: #f59e0b;
            background: rgba(245, 158, 11, 0.06);
        }
        .indicator-info {
            border-left-color: #38bdf8;
            background: rgba(56, 189, 248, 0.06);
        }
        .indicator-low {
            border-left-color: #10b981;
            background: rgba(16, 185, 129, 0.06);
        }

        /* Form Container */
        .form-section-header {
            font-size: 0.82rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: #38bdf8;
            margin: 18px 0 8px 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.07);
            padding-bottom: 4px;
        }
        
        /* Preset Button Cards */
        div[data-testid="stHorizontalBlock"] .stButton > button {
            width: 100%;
            border-radius: 8px;
            font-weight: 600;
            font-size: 0.82rem;
            transition: all 0.2s ease;
            border: 1px solid rgba(255, 255, 255, 0.12);
            background: #111a2e;
            color: #e2e8f0;
        }
        div[data-testid="stHorizontalBlock"] .stButton > button:hover {
            border-color: #00f2fe;
            color: #00f2fe;
            box-shadow: 0 0 12px rgba(0, 242, 254, 0.25);
        }

        /* Submit Predict Button */
        button[kind="primaryFormSubmit"], div.stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #0284c7 0%, #00f2fe 100%) !important;
            color: #041322 !important;
            font-weight: 800 !important;
            font-size: 1.05rem !important;
            letter-spacing: 0.05em !important;
            border: none !important;
            padding: 12px 24px !important;
            border-radius: 10px !important;
            box-shadow: 0 4px 20px rgba(0, 242, 254, 0.35) !important;
            transition: all 0.25s ease !important;
        }
        button[kind="primaryFormSubmit"]:hover, div.stButton > button[kind="primary"]:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 25px rgba(0, 242, 254, 0.5) !important;
        }

        /* Filter Panel Box */
        .filter-panel {
            background: #0f172a;
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 12px;
            padding: 16px 20px;
            margin-bottom: 20px;
        }

        /* Console Footer */
        .finsec-footer {
            margin-top: 50px;
            padding: 20px 0;
            border-top: 1px solid rgba(255, 255, 255, 0.07);
            text-align: center;
            font-size: 0.75rem;
            color: #64748b;
            letter-spacing: 0.03em;
        }

        /* Dual Meter Grid */
        .dual-meter-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }

        /* ================= MOBILE & TABLET RESPONSIVENESS ================= */
        @media (max-width: 992px) {
            .main .block-container {
                padding-top: 1rem !important;
                padding-left: 1rem !important;
                padding-right: 1rem !important;
                padding-bottom: 2.5rem !important;
            }
            .academic-banner {
                padding: 16px 18px !important;
            }
            .academic-title {
                font-size: 1.25rem !important;
            }
            .metric-value {
                font-size: 1.5rem !important;
            }
        }

        @media (max-width: 768px) {
            /* Full mobile responsive layout */
            .main .block-container {
                padding: 0.8rem 0.6rem 2rem 0.6rem !important;
            }
            
            /* Banner collapses to mobile card */
            .academic-banner {
                flex-direction: column !important;
                align-items: flex-start !important;
                padding: 14px 16px !important;
                gap: 10px !important;
            }
            .academic-title {
                font-size: 1.15rem !important;
                line-height: 1.3 !important;
            }
            .academic-subtitle {
                font-size: 0.78rem !important;
            }
            .academic-badge {
                align-self: flex-start !important;
                font-size: 0.68rem !important;
                padding: 4px 10px !important;
            }

            /* Metric / KPI Cards on mobile */
            .metric-card {
                padding: 14px 16px !important;
                margin-bottom: 10px !important;
            }
            .metric-title {
                font-size: 0.68rem !important;
            }
            .metric-value {
                font-size: 1.35rem !important;
            }
            .metric-sub {
                font-size: 0.7rem !important;
            }

            /* Dual Meter Grid collapses to single column on mobile */
            .dual-meter-grid {
                grid-template-columns: 1fr !important;
                gap: 12px !important;
            }

            /* Risk Assessment Container */
            .risk-assessment-container {
                padding: 16px 14px !important;
            }
            .badge-low, .badge-medium, .badge-high {
                font-size: 0.95rem !important;
                padding: 6px 12px !important;
            }

            /* Tables with smooth horizontal touch scroll */
            div[data-testid="stDataFrame"], div[data-testid="stTable"], .dataframe {
                width: 100% !important;
                overflow-x: auto !important;
                -webkit-overflow-scrolling: touch !important;
            }

            /* Buttons and Form inputs touch-friendly */
            div[data-testid="stHorizontalBlock"] .stButton > button {
                font-size: 0.8rem !important;
                padding: 10px 8px !important;
                margin-bottom: 8px !important;
            }
            button[kind="primaryFormSubmit"], div.stButton > button[kind="primary"] {
                font-size: 0.95rem !important;
                padding: 10px 16px !important;
                width: 100% !important;
            }

            /* Indicator Explainability Boxes */
            .indicator-box {
                padding: 10px 14px !important;
            }

            /* Footer */
            .finsec-footer {
                margin-top: 30px !important;
                font-size: 0.7rem !important;
                padding: 15px 0 !important;
            }
        }

        @media (max-width: 480px) {
            .academic-title {
                font-size: 1.05rem !important;
            }
            .metric-value {
                font-size: 1.2rem !important;
            }
            .brand-name {
                font-size: 1.15rem !important;
            }
        }
    </style>
    """
