import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# Funkcja do pobrania spółek z indeksu WIG
@st.cache_data
def get_wig_tickers():
    # Pobranie składników indeksu WIG
    wig_constituents = investpy.get_index_stocks(index='WIG', country='poland')
    return wig_constituents['symbol'].tolist()

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

# Funkcje do analizy portfela
def portfolio_performance(weights, mean_returns, cov_matrix):
    returns = np.sum(weights * mean_returns) * 252
    risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    return returns, risk


def random_portfolios(num_portfolios, mean_returns, cov_matrix):
    results = np.zeros((3, num_portfolios))
    weights_record = []
    
    for _ in range(num_portfolios):
        weights = np.random.random(len(mean_returns))
        weights /= np.sum(weights)
        weights_record.append(weights)
        
        portfolio_return, portfolio_risk = portfolio_performance(weights, mean_returns, cov_matrix)
        
        results[0, _] = portfolio_return
        results[1, _] = portfolio_risk
        results[2, _] = _  # Indeks portfolio
    
    return results, weights_record

# Główna aplikacja Streamlit
st.title("Aplikacja do budowy portfela inwestycyjnego (Indeks WIG)")

# Pobranie listy spółek z WIG
try:
    tickers  = [
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

    selected_tickers = st.multiselect(
        "Wybierz spółki do analizy:",
        tickers,
        default=tickers[:5]  # Domyślnie wybierane pierwsze 5 spółek
    )
except Exception as e:
    st.error(f"Nie udało się pobrać składników indeksu WIG: {e}")
    tickers = []
    selected_tickers = []

start_date = st.date_input("Data początkowa:", value=pd.to_datetime("2018-01-01"))
end_date = st.date_input("Data końcowa:", value=pd.to_datetime("2023-12-31"))

if selected_tickers:
    # Pobranie danych
    try:
        data = fetch_data(selected_tickers, start_date, end_date)
        
        # Wyświetlanie danych
        st.subheader("Podgląd danych cenowych")
        st.write(data.tail())

        # Obliczanie stóp zwrotu
        returns = data.pct_change().dropna()

        # Statystyki spółek
        st.subheader("Statystyki spółek")
        mean_returns = returns.mean() * 252
        volatilities = returns.std() * np.sqrt(252)
        stats = pd.DataFrame({
            "Średni zwrot roczny": mean_returns,
            "Ryzyko roczne": volatilities
        })
        st.write(stats)

        # Optymalizacja portfela
        cov_matrix = returns.cov()
        num_portfolios = 10000

        results, weights_record = random_portfolios(num_portfolios, mean_returns, cov_matrix)

        # Ustalanie poziomu ryzyka przez inwestora
        st.subheader("Ustal swój poziom ryzyka")
        max_risk = st.slider(
            "Maksymalne ryzyko (odchylenie standardowe):",
            min_value=float(results[1, :].min()),
            max_value=float(results[1, :].max()),
            value=float(results[1, :].mean())
        )

        # Filtracja portfeli na podstawie ryzyka
        valid_portfolios = results[:, results[1, :] <= max_risk]
        if valid_portfolios.shape[1] > 0:
            best_portfolio_idx = np.argmax(valid_portfolios[0])
            best_portfolio = valid_portfolios[:, best_portfolio_idx]
            best_weights = weights_record[int(best_portfolio[2])]

            # Wizualizacja granicy efektywnej
            st.subheader("Granica efektywna portfela")
            plt.figure(figsize=(10, 6))
            plt.scatter(results[1, :], results[0, :], c=results[0, :] / results[1, :], cmap='viridis')
            plt.colorbar(label='Sharpe Ratio')
            plt.scatter(best_portfolio[1], best_portfolio[0], marker='*', color='r', s=200, label='Najlepszy portfel')
            plt.title('Efektywna granica portfela')
            plt.xlabel('Ryzyko (odchylenie standardowe)')
            plt.ylabel('Zwrot roczny')
            plt.legend(labelspacing=0.8)
            st.pyplot(plt)

            # Wyświetlenie szczegółów portfela
            st.subheader("Wybrany portfel")
            allocation = pd.DataFrame({
                "Spółka": selected_tickers,
                "Udział w portfelu": [f"{weight:.2%}" for weight in best_weights]
            })
            st.write(f"Zwrot: {best_portfolio[0]:.2f}")
            st.write(f"Ryzyko: {best_portfolio[1]:.2f}")
            st.write(allocation)
        else:
            st.warning("Żaden portfel nie spełnia określonego poziomu ryzyka.")
    except KeyError as e:
        st.error(f"Wystąpił błąd podczas pobierania danych: {e}")
else:
    st.write("Wybierz co najmniej jedną spółkę, aby rozpocząć analizę.")