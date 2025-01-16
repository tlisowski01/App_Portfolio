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
    portfolios = {
        "returns": [],
        "risks": [],
        "weights": []
    }
    for _ in range(num_portfolios):
        weights = np.random.random(len(mean_returns))
        weights /= np.sum(weights)
        portfolio_return, portfolio_risk = portfolio_performance(weights, mean_returns, cov_matrix)
        portfolios["returns"].append(portfolio_return)
        portfolios["risks"].append(portfolio_risk)
        portfolios["weights"].append(weights)
    return portfolios

# Główna aplikacja Streamlit
st.title("Efektywna granica portfela (Markowitza)")

# Wybór akcji
tickers = ["PKN.WA", "JSW.WA", "ALE.WA", "KGH.WA", "BMC.WA"]  # Przykładowe tickery
selected_tickers = st.multiselect("Wybierz spółki do analizy:", tickers, default=tickers)

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

        # Generowanie portfeli
        num_portfolios = 10000
        portfolios = generate_portfolios(num_portfolios, mean_returns, cov_matrix)

        # Wizualizacja efektywnej granicy
        st.subheader("Granica efektywna portfela")
        plt.figure(figsize=(10, 6))
        plt.scatter(portfolios["risks"], portfolios["returns"], c=np.array(portfolios["returns"]) / np.array(portfolios["risks"]), cmap="viridis")
        plt.colorbar(label="Sharpe Ratio")
        plt.title("Efektywna granica portfela")
        plt.xlabel("Ryzyko (odchylenie standardowe)")
        plt.ylabel("Zwrot roczny (%)")
        st.pyplot(plt)

        # Ustalanie poziomu ryzyka
        st.subheader("Przeglądaj portfele dla wybranego poziomu ryzyka")
        max_risk = st.slider(
            "Maksymalne ryzyko (odchylenie standardowe):",
            min_value=float(min(portfolios["risks"])),
            max_value=float(max(portfolios["risks"])),
            value=float(np.mean(portfolios["risks"]))
        )

        # Wybór portfela na podstawie ryzyka
        valid_indices = [i for i, risk in enumerate(portfolios["risks"]) if risk <= max_risk]
        if valid_indices:
            best_idx = max(valid_indices, key=lambda i: portfolios["returns"][i])
            best_return = portfolios["returns"][best_idx]
            best_risk = portfolios["risks"][best_idx]
            best_weights = portfolios["weights"][best_idx]

            # Szczegóły portfela
            st.subheader("Wybrany portfel")
            allocation = pd.DataFrame({
                "Spółka": selected_tickers,
                "Udział w portfelu (%)": [f"{weight * 100:.2f}%" for weight in best_weights]
            })
            st.write(f"Zwrot (%): {best_return * 100:.2f}")
            st.write(f"Ryzyko (%): {best_risk * 100:.2f}")
            st.write(allocation)
        else:
            st.warning("Żaden portfel nie spełnia określonego poziomu ryzyka.")
    except KeyError as e:
        st.error(f"Wystąpił błąd podczas pobierania danych: {e}")
else:
    st.write("Wybierz co najmniej jedną spółkę, aby rozpocząć analizę.")
