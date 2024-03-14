import matplotlib.patches as patches
import yfinance as yf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from src.train import run_GOLEMTS
from src.train import run_DYNOTEARS
from src.train import run_DAGMATS

def snp100():
    df, col, d, tickers= get100() 

    #print("dynotears")
    #dynotears = run_DYNOTEARS(99, d, 1, df, epochs=1)
    print("golem")
    golemts = run_GOLEMTS(99, d, 1, df)
    print("DAGMATS")
    dagmats = run_DAGMATS(99, d, 1, df, epochs=1000)

    #display(dynotears, col, "dynotears", tickers)
    display(golemts, col, "golemts", tickers)
    display(dagmats, col, "dagmats", tickers)

def get100():
    xtickers = "COST CMCSA SBUX F MCD LOW HD CL PM MO PEP KO CVX OXY COP MET ALL MS AIG USB BLK BAC LLY MRK ABT PFE BIIB JNJ BMY DD UPS GD MMM HON LMT QCOM V NVDA ADBE GOOGL MA NFLX IBM VZ DUK NEE".split(" ")
    ytickers = "DIS NKE CHTR AMZN TGT GM BKNG PG MDLZ WBA WMT KMI SLB XOM BK C JPM SPG AXP WFC COF GS AMGN ABBV MDT CVS DHR UNH GILD EMR CAT GE UNP BA FDX TXN AAPL ACN CSCO ORCL GOOG MSFT INTC T EXC SO".split(" ")
    # Read and print the stock tickers that make up S&P500
    tickers = [val for pair in zip(xtickers, ytickers) for val in pair]

    # Get the data for this tickers from yahoo finance
    data = yf.download(tickers,'2021-1-1','2021-7-12', auto_adjust=True)['Close'][:100]

    data.reset_index(inplace=True, drop=True)
    current = data.iloc[0]
    dataln = np.log(current/data)

    data_norm = dataln - dataln.mean()
    data_norm = data_norm/data_norm.std()
    col = data_norm.columns

    curr = data_norm.drop(99, axis=0)
    pre = data_norm.drop(0, axis=0)

    Y = data_norm.to_numpy()
    Yt = np.append(Y[:99], Y[1:], axis=1)
    
    return Yt, col, Y.shape[1], tickers

def display(df, col, name, tickers):
    df = pd.DataFrame(df[:92])
    
    df.columns = col
    df = df[tickers]
    df = df.transpose()
    df.columns = col
    df = df[tickers]
    df.transpose()

    sns.set(font_scale=.6)
    ax = sns.heatmap(abs(df),  cmap='Reds',linewidths=.05);
    ax.set_xlabel('', fontsize=10)
    ax.set_ylabel('', fontsize=10)

    ax.add_patch(
     patches.Rectangle(
         (0, 0),
         14,
         14,
         edgecolor='black',
         fill=False,
         lw=.5
     ) )

    ax.text(15, 7, "Consumer Cyclicals")

    ax.add_patch(
     patches.Rectangle(
         (14, 14),
         23-14,
         23-14,
         edgecolor='black',
         fill=False,
         lw=.5
     ) )


    ax.text(24, 20, "Consumer Non-Cyclicals")

    ax.add_patch(
     patches.Rectangle(
         (23, 23),
         29-23,
         29-23,
         edgecolor='black',
         fill=False,
         lw=.5
     ) )


    ax.text(30, 27, "Energy")

    ax.add_patch(
     patches.Rectangle(
         (29, 29),
         44-29,
         44-29,
         edgecolor='black',
         fill=False,
         lw=.5
     ) )

    ax.text(45, 37, "Financials")

    ax.add_patch(
     patches.Rectangle(
         (44, 44),
         58-44,
         58-44,
         edgecolor='black',
         fill=False,
         lw=.5
     ) )

    ax.text(59, 52, "Healthcare")

    ax.add_patch(
     patches.Rectangle(
         (58, 58),
         70-58,
         70-58,
         edgecolor='black',
         fill=False,
         lw=.5
     ) )

    ax.text(71, 64, "Industrials")

    ax.add_patch(
     patches.Rectangle(
         (70, 70),
         87-70,
         87-70,
         edgecolor='black',
         fill=False,
         lw=.5
     ) )

    ax.text(58, 79, "Technology")


    ax.add_patch(
     patches.Rectangle(
         (87, 87),
         5,
         5,
         edgecolor='black',
         fill=False,
         lw=.5
     ) )

    ax.text(79, 90, "Utilities")

    print("saving snp100 to images")
    plt.savefig('images/stocks_{}.png'.format(name), dpi=500)
