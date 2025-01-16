import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# Funkcja do pobrania danych
@st.cache_data
def fetch_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)
    return data["Adj Close"] if "Adj Close" in data.columns else data["Close"]

# Funkcja do obliczania wyników portfela
def portfolio_performance(weights, mean_returns, cov_matrix):
    returns = np.sum(weights * mean_returns) * 252
    risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    return returns, risk

# Funkcja do generowania portfeli
def generate_portfolios(mean_returns, cov_matrix, num_portfolios=10000):
    results = []
    for _ in range(num_portfolios):
        weights = np.random.random(len(mean_returns))
        weights /= np.sum(weights)
        returns, risk = portfolio_performance(weights, mean_returns, cov_matrix)
        results.append({"returns": returns, "risks": risk, "weights": weights})
    return results

# Lista dostępnych spółek
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

# Streamlit aplikacja
st.title("Aplikacja do budowy portfela inwestycyjnego")

# Wybór spółek
selected_tickers = st.multiselect(
    "Wybierz spółki do analizy:",
    tickers,
    default=["PKN.WA", "JSW.WA", "ALE.WA", "KGH.WA", "BMC.WA"]
)

# Data początkowa i końcowa
start_date = st.date_input("Data początkowa:", value=pd.to_datetime("2018-01-01"))
end_date = st.date_input("Data końcowa:", value=pd.to_datetime("2023-12-31"))

# Pobieranie danych
if selected_tickers:
    data = fetch_data(selected_tickers, start_date, end_date)
    returns = data.pct_change().dropna()
    mean_returns = returns.mean()
    cov_matrix = returns.cov()

    # Generowanie portfeli
    portfolios = generate_portfolios(mean_returns, cov_matrix)

    st.subheader("Granica efektywna portfela")
    plt.figure(figsize=(10, 6))
    risks = [p["risks"] for p in portfolios]
    returns = [p["returns"] for p in portfolios]
    plt.scatter(risks, returns, c=[r / ri for r, ri in zip(returns, risks)], cmap="viridis")
    plt.colorbar(label="Sharpe Ratio")
    plt.xlabel("Ryzyko (odchylenie standardowe)")
    plt.ylabel("Zwrot roczny")
    st.pyplot(plt)

    # Wybór poziomu ryzyka
    st.subheader("Wybierz poziom ryzyka")
    max_risk = st.slider(
        "Maksymalne ryzyko (odchylenie standardowe):",
        min_value=min(risks),
        max_value=max(risks),
        value=np.mean(risks)
    )

    # Filtracja portfeli na podstawie poziomu ryzyka
    filtered_portfolios = [p for p in portfolios if p["risks"] <= max_risk]
    if filtered_portfolios:
        best_portfolio = max(filtered_portfolios, key=lambda p: p["returns"])

        # Wyświetlanie wyników najlepszego portfela
        allocation = pd.DataFrame({
            "Spółka": selected_tickers,
            "Udział w portfelu (%)": [f"{w * 100:.2f}%" for w in best_portfolio["weights"]]
        })
        st.write(f"Zwrot roczny (%): {best_portfolio['returns'] * 100:.2f}")
        st.write(f"Ryzyko roczne (%): {best_portfolio['risks'] * 100:.2f}")
        st.write(allocation)

        # Analiza wpływu na portfel
        st.subheader("Analiza wpływu spółek na portfel")
        negative_impact = []
        for ticker, weight in zip(selected_tickers, best_portfolio["weights"]):
            temp_weights = np.array(best_portfolio["weights"])
            temp_weights[selected_tickers.index(ticker)] = 0  # ustawiamy wagę spółki na 0
            temp_return, temp_risk = portfolio_performance(temp_weights, mean_returns, cov_matrix)
            negative_impact.append({"spółka": ticker, "impact": best_portfolio["risks"] - temp_risk})

        # Pokazanie spółki, która najbardziej wpływa na ryzyko portfela
        negative_impact_sorted = sorted(negative_impact, key=lambda x: x["impact"], reverse=True)
        most_impactful = negative_impact_sorted[0]
        st.write(f"Spółka, która najbardziej wpływa na ryzyko portfela: {most_impactful['spółka']} "
                 f"(Zmiana ryzyka: {most_impactful['impact'] * 100:.2f}%)")
    else:
        st.warning("Żaden portfel nie spełnia określonego poziomu ryzyka.")
else:
    st.warning("Wybierz co najmniej jedną spółkę do analizy.")
