"""
Functions that return the universe name and ticker list within it
"""
def dow_30() -> tuple[list, str] :
    DOW30 = ["AAPL","AMGN","AMZN","AXP","BA","CAT","CRM","CSCO","CVX","DIS","GS","HD","HON","IBM","JNJ","JPM","KO","MCD","MMM","MRK","MSFT","NKE","NVDA","PG","SHW","TRV","UNH","V","VZ","WMT"]
    name = "DOW30"
    return DOW30, name