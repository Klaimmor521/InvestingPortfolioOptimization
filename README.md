# InvestingPortfolioOptimization
## Investing diversification for optimizing the portfolio

## Installation:
1. Create folder for the project and run there in terminal `git clone MY_URL_REPOSITORY`
2. In the project run `python -m venv venv`
3. Activate virtual space in the terminal `venv\Scripts\Activate.ps1`. To deactivate virtual space use `deactivate`
4. Install the dependencies `pip install -r requirements.txt`
5. Run app in the terminal `python main_app.py`

And... **Viola!** You can use this project!

## I. General Description of the Project and its Purpose
The Portfolio Optimizer project is a desktop application designed to help users (students, novice investors) understand and apply the principles of diversification when forming an investment portfolio.

**The purpose of the application:**
Visually demonstrate **how to optimize a portfolio of selected stocks based on their historical data**. The application calculates and visualizes various asset combinations, striving to find the optimal ratio between expected return and risk, using the classical Markowitz portfolio theory. The main task is not to predict future market behavior, but to show how diversification could work based on past data, and provide the user with a tool for making more informed investment decisions.

## II. How The App Works
1. **Receiving Input Data**: The user enters a list of tickers of the stocks they are interested in, the desired historical period for analysis, and the annual risk-free rate.
2. **Data Download**: Using the Yahoo Finance API, the program downloads historical daily quotes (closing prices) for the specified tickers for the selected period.
3. **Calculation of Returns and Statistics**: Based on the prices received, daily returns on assets are calculated. Then the key statistical indicators are calculated: annualized average returns (as an estimate of expected returns) and an annualized covariance matrix of returns (reflecting risk and the relationship of assets).
4. **Markowitz Model optimization**: Using mathematical optimization methods (SciPy library), the program solves the following tasks:
    - **Building the Efficiency Boundary**: There are many portfolios, each of which offers the maximum expected return for a given level of risk (or the minimum risk for a given return).
    - **Definition of Key Portfolios**:
        - **Minimum Variance Portfolio (MVP)**: The portfolio with the lowest possible risk of available assets.
        - **Maximum Sharpe Ratio (MSR) Portfolio**: A portfolio offering the best return-to-risk ratio (taking into account the risk-free rate).
5. **Visualization and Display**: The calculation results are presented to the user in graphical (Efficiency Boundary) and textual (composition and characteristics of MVP and MSR) form.
6. **Automatic Saving**: The main query parameters and results for MVP and MSR are automatically saved to a local JSON file (last_optimization_results.json) for later analysis.

## III. Instructions for Using the Interface
1. **The "Enter stock tickers separated by commas:" field**:
    - Here you need to enter the stock tickers that you want to include in the analysis. Tickers must match the designations on Yahoo Finance (for example, AAPL for Apple, GOOG for Alphabet, MSFT for Microsoft).
    - Enter the tickers separated by commas, without spaces before or after the comma (the spaces around the tickers themselves will be deleted automatically).
    - It is recommended to use from 2 to 10 tickers for clarity and speed of calculations.
2. **The field "Start date (YYYY-MM-DD)**":
    - Specify the starting date of the historical period for which the data will be uploaded. Format: Year-Month-Day (for example, 2020-01-01).
    - **The date should not be in the future**.
3. **Field "End date (YYYY-MM-DD)"**:
    - Specify the end date of the historical period. Format: Year-Month-Day (for example, 2023-12-31).
    - The date must not be earlier than the starting date. If a current or future date is specified, it will be automatically adjusted to yesterday.
    - The recommended minimum analysis period is at least 1-2 years to obtain more stable statistical estimates.
4. **The Risk-free field. rate (% per annum)"**:
    - Enter the annual risk-free percentage rate (for example, 2.0 for 2%). This value is used to calculate the Sharpe coefficient.
    - Usually, the yield on government bonds is used as a risk-free rate.
5. **The "Calculate Portfolio" button**:
    - After entering all the data, click this button to start the data loading and optimization process.
6. **Status area (under the button)**:
    - Progress messages will be displayed here: loading data, starting optimization, error messages or warnings (for example, if not all tickers were loaded), and a message about successful completion and saving.
7. **Graph Area (bottom left)**:
    - After successful calculation, the "Portfolio Performance Boundary" graph will appear here.
    - **Axes**: Horizontal – "Volatility (Annual Standard Deviation)" (a measure of risk), Vertical – "Expected Return (Annual)".
    - **Lines and dots**:
        - **Blue/Blue curve**: Efficiency Boundary – shows the best possible combinations of risk and return.
        - **Red Circle (MVP)**: Minimum Variance Portfolio.
        - **Green Star (MSR)**: A portfolio with a Maximum Sharpe Ratio.
        - **Multi-colored crosses**: The position of individual stocks (if you had invested 100% in one of them).
    - **Legend**: Explains the notation on the graph.
    - **Matplotlib toolbar (under the graph)**: Allows you to scale the graph, move it, save the graph image, etc.
8. **The "Optimization Results" area (bottom right)**:
    - Detailed specifications for:
        - **Minimum Variance Portfolio (MVP)**: Expected return, Volatility (Risk), Portfolio composition (which stocks and in what proportion (in %) are recommended to be included).
        - **Briefcase Max. Coefficient. Sharpe Ratio (MSR)**: Expected Return, Volatility (Risk), Sharpe Ratio, Portfolio Composition.
    - Only assets with a weight of more than 0.01% are displayed in the portfolio (very small weights are omitted for readability).

## IV. How to Interpret and Use Optimization Results
The results provided by the app will help you understand how funds could be distributed among selected stocks to achieve various investment goals based on their past behavior.

1. **The Limit of Efficiency**:
    - Meaning: Any portfolio lying on this curve is "efficient" – you cannot get more returns with the same level of risk, or less risk with the same return, using only selected assets. Portfolios below the curve are ineffective.
    - Usage: Look at the shape of the curve. It shows how the risk increases when trying to get a higher return. You can choose a point on this curve that matches your individual risk appetite. Someone will prefer a point closer to the lower left corner (less risk, less return), someone to the right and higher.
2. **Minimum Variance Portfolio (MVP)**:
    - Meaning: This is the most "safe" (in terms of volatility) portfolio that can be made up of these stocks. It's not necessarily the most profitable one.
    - Usage: If your main goal is to minimize possible fluctuations in the value of the portfolio, pay attention to the composition of the MVP.
3. **Portfolio with Maximum Sharpe Ratio (MSR)**:
    - Meaning: The Sharpe ratio shows how much return you receive above the risk-free rate per unit of accepted risk. An MSR is a portfolio that has historically provided the best compensation for risk. It is often considered the theoretically "optimal" risk portfolio.
    - Usage: If you are looking for a balanced portfolio with a good return–to-risk ratio, MSR is a good candidate. You can combine this portfolio with risk-free assets (such as bonds) to achieve the desired overall risk level of your entire capital.
4. **Portfolio Composition (Weights)**:
    - Meaning: The percentages show how much of the total investment amount is recommended to invest in each share to form an MVP or MSR.
    - Using: This is a practical recommendation. If the model offers 60% in AAPL and 40% in GOOG, this means that out of 1,000 conventional units, 600 need to be invested in AAPL, and 400 in GOOG.
5. **Comparison with Individual Stocks**:
    - Look at the graph where the points of individual stocks are compared to the Performance Boundary. It often turns out that a diversified portfolio (a point on the border) can offer the same return as a single stock, but with less risk, or a higher return with the same risk. **This is the effect of diversification.**

# Important when using the results:
**This is not a prediction**: The results are based on the past. The future may be different.
**This is not an investment recommendation**: The app is an educational tool. Real investment decisions should be made taking into account many other factors (your financial goals, the investment period, the current market situation, fundamental analysis of companies, etc.) and, possibly, with the advice of a financial adviser.
**Periodic rebalancing**: If you decide to follow any of the proposed portfolios, keep in mind that asset weights will change over time due to different price dynamics. Optimal portfolios require periodic rebalancing (returning the weights to the target ones).