import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# Funkcja do pobrania spółek z indeksu WIG
@st.cache_data
def get_wig_tickers():
    return ["PKN.WA", "JSW.WA", "ALE.WA", "KGH.WA", "BMC.WA"]  # Przykładowe tickery

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
    returns = np.sum(weights * mean_returns)
    risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return returns, risk

def random_portfolios(num_portfolios, mean_returns, cov_matrix):
    results = np.zeros((2, num_portfolios))  # Dla zwrotu i ryzyka
    weights_record = []
    
    for _ in range(num_portfolios):
        weights = np.random.random(len(mean_returns))
        weights /= np.sum(weights)
        weights_record.append(weights)
        
        portfolio_return, portfolio_risk = portfolio_performance(weights, mean_returns, cov_matrix)
        
        results[0, _] = portfolio_return
        results[1, _] = portfolio_risk
    
    return results, weights_record

# Główna aplikacja Streamlit
st.title("Aplikacja do budowy portfela inwestycyjnego (Indeks WIG)")

# Pobranie listy spółek z WIG
tickers = get_wig_tickers()
selected_tickers = st.multiselect(
    "Wybierz spółki do analizy:",
    tickers,
    default=tickers[:5]  # Domyślnie wybierane pierwsze 5 spółek
)

start_date = st.date_input("Data początkowa:", value=pd.to_datetime("2018-01-01"))
end_date = st.date_input("Data końcowa:", value=pd.to_datetime("2023-12-31"))

if selected_tickers:
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
            "Średni zwrot roczny (%)": mean_returns * 100,
            "Ryzyko roczne (%)": volatilities * 100
        })
        st.write(stats)

        # Optymalizacja portfela
        cov_matrix = returns.cov()
        num_portfolios = 10000

        # Generowanie portfeli
        results, weights_record = random_portfolios(num_portfolios, mean_returns, cov_matrix)

        # Wizualizacja granicy efektywnej
        st.subheader("Granica efektywna portfela")
        plt.figure(figsize=(10, 6))
        plt.scatter(results[1, :], results[0, :], c=results[0, :] / results[1, :], cmap='viridis')
        plt.colorbar(label='Sharpe Ratio')
        plt.title('Efektywna granica portfela')
        plt.xlabel('Ryzyko (odchylenie standardowe)')
        plt.ylabel('Zwrot roczny (%)')
        st.pyplot(plt)

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
            best_weights = weights_record[int(np.where(results[1, :] == best_portfolio[1])[0][0])]

            # Wyświetlenie szczegółów portfela
            st.subheader("Wybrany portfel")
            allocation = pd.DataFrame({
                "Spółka": selected_tickers,
                "Udział w portfelu (%)": [f"{weight * 100:.2f}%" for weight in best_weights]
            })
            st.write(f"Zwrot (%): {best_portfolio[0] * 100:.2f}")
            st.write(f"Ryzyko (%): {best_portfolio[1] * 100:.2f}")
            st.write(allocation)
        else:
            st.warning("Żaden portfel nie spełnia określonego poziomu ryzyka.")
    except KeyError as e:
        st.error(f"Wystąpił błąd podczas pobierania danych: {e}")
else:
    st.write("Wybierz co najmniej jedną spółkę, aby rozpocząć analizę.")
