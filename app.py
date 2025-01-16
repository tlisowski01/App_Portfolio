import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# Funkcja do pobrania danych
@st.cache_data
def fetch_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)
    if "Adj Close" in data.columns:
        return data["Adj Close"]
    elif "Close" in data.columns:
        return data["Close"]
    else:
        raise KeyError("Brak odpowiednich danych w pobranym zbiorze.")

# Funkcja do analizy portfela
def portfolio_performance(weights, mean_returns, cov_matrix):
    returns = np.sum(weights * mean_returns)
    risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return returns, risk

def generate_portfolios(num_portfolios, mean_returns, cov_matrix):
    portfolios = []
    for _ in range(num_portfolios):
        weights = np.random.random(len(mean_returns))
        weights /= np.sum(weights)
        portfolio_return, portfolio_risk = portfolio_performance(weights, mean_returns, cov_matrix)
        portfolios.append({
            "returns": portfolio_return,
            "risks": portfolio_risk,
            "weights": weights
        })
    return portfolios

# Funkcja do rysowania wykresu efektywnej granicy
def plot_efficient_frontier(portfolios):
    risks = [p["risks"] for p in portfolios]
    returns = [p["returns"] for p in portfolios]
    plt.figure(figsize=(10, 6))
    plt.scatter(risks, returns, c=np.array(returns) / np.array(risks), cmap="viridis")
    plt.colorbar(label="Sharpe Ratio")
    plt.title("Efektywna granica portfela")
    plt.xlabel("Ryzyko (odchylenie standardowe)")
    plt.ylabel("Zwrot roczny (%)")
    st.pyplot(plt)

# Główna aplikacja Streamlit
st.title("Efektywna granica portfela (Markowitza)")

# Pełna lista tickerów
tickers = [
    "PKN.WA", "JSW.WA", "ALE.WA", "KGH.WA", "BMC.WA", "CDR.WA", "XTB.WA",
    "PCO.WA", "ENI.WA", "ZAB.WA", "MLS.WA", "PZU.WA", "RFK.WA", "CCC.WA",
    "11B.WA", "CPS.WA", "TXT.WA", "DNP.WA", "PKO.WA", "SNT.WA", "PGE.WA",
    "LPP.WA", "KCH.WA", "PXM.WA", "LWB.WA", "PEO.WA", "DAT.WA", "KRU.WA",
    "BDX.WA", "LBW.WA", "GRX.WA", "EUR.WA", "ASB.WA", "APR.WA", "PKP.WA",
    "ATT.WA", "KTY.WA", "TPE.WA", "MAB.WA", "CIG.WA", "PUR.WA", "MRC.WA",
    "RBW.WA", "MRB.WA", "CRI.WA", "XTP.WA", "ENG.WA", "ALR.WA", "KER.WA",
    "OPL.WA", "ENA.WA", "TEN.WA", "MBR.WA", "HUG.WA", "CLC.WA", "ABE.WA",
    "4MS.WA", "MBK.WA", "ELT.WA", "GPW.WA", "HRS.WA", "EAT.WA", "RVU.WA",
    "MSW.WA", "PLW.WA", "DIG.WA", "TRK.WA", "BFT.WA", "SVE.WA", "BDZ.WA",
    "OND.WA", "BIO.WA", "CBF.WA", "AST.WA", "SPL.WA", "MIL.WA", "VOX.WA",
    "CLN.WA", "BHW.WA", "CAR.WA", "ATC.WA", "ING.WA", "MLK.WA", "ACP.WA",
    "MDG.WA", "VRC.WA", "WLT.WA", "PEP.WA", "CRJ.WA", "GEA.WA", "ONO.WA",
    "COG.WA", "VVD.WA", "ABS.WA", "WTN.WA", "LES.WA", "BOS.WA", "NNG.WA",
    "GMT.WA", "DVL.WA"
]

# Domyślnie wybrane tickery
default_tickers = ["PKN.WA", "JSW.WA", "ALE.WA", "KGH.WA", "BMC.WA"]

# Wybór akcji
selected_tickers = st.multiselect("Wybierz spółki do analizy:", tickers, default=default_tickers)

start_date = st.date_input("Data początkowa:", value=pd.to_datetime("2018-01-01"))
end_date = st.date_input("Data końcowa:", value=pd.to_datetime("2023-12-31"))

if selected_tickers:
    try:
        # Pobranie danych
        data = fetch_data(selected_tickers, start_date, end_date)

        # Obliczenie stóp zwrotu i statystyk
        returns = data.pct_change().dropna()
        mean_returns = returns.mean() * 252
        cov_matrix = returns.cov() * 252

        # Generowanie portfeli (tylko raz)
        if "portfolios" not in st.session_state:
            num_portfolios = 10000
            st.session_state.portfolios = generate_portfolios(num_portfolios, mean_returns, cov_matrix)

        # Wyświetlenie wykresu efektywnej granicy
        st.subheader("Granica efektywna portfela")
        plot_efficient_frontier(st.session_state.portfolios)

        # Ustalanie poziomu ryzyka
        st.subheader("Przeglądaj portfele dla wybranego poziomu ryzyka")
        max_risk = st.slider(
            "Maksymalne ryzyko (odchylenie standardowe):",
            min_value=float(min(p["risks"] for p in st.session_state.portfolios)),
            max_value=float(max(p["risks"] for p in st.session_state.portfolios)),
            value=float(np.mean([p["risks"] for p in st.session_state.portfolios]))
        )

        # Wybór portfela na podstawie poziomu ryzyka
        valid_portfolios = [p for p in st.session_state.portfolios if p["risks"] <= max_risk]
        if valid_portfolios:
            best_portfolio = max(valid_portfolios, key=lambda p: p["returns"])
            best_return = best_portfolio["returns"]
            best_risk = best_portfolio["risks"]
            best_weights = best_portfolio["weights"]

            # Szczegóły portfela
            st.subheader("Wybrany portfel")
            allocation = pd.DataFrame({
                "Spółka": selected_tickers,
                "Udział w portfelu (%)": [f"{weight * 100:.2f}%" for weight in best_weights]
            })
            st.write(f"Zwrot roczny (%): {best_return * 100:.2f}")
            st.write(f"Ryzyko roczne (%): {best_risk * 100:.2f}")
            st.write(allocation)
        else:
            st.warning("Żaden portfel nie spełnia określonego poziomu ryzyka.")
    except KeyError as e:
        st.error(f"Wystąpił błąd podczas pobierania danych: {e}")
else:
    st.write("Wybierz co najmniej jedną spółkę, aby rozpocząć analizę.")
