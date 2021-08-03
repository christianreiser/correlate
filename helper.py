import seaborn as sns
from matplotlib import pyplot as plt


def histograms(df, save_path):
    # plot distributions
    for attribute in df.columns:
        print(attribute)

        sns.set(style="ticks")

        x = df[attribute]#.to_numpy()

        f, (ax_box, ax_hist) = plt.subplots(2, sharex=True,
                                            gridspec_kw={"height_ratios": (.15, .85)})

        sns.boxplot(x=x, ax=ax_box, showmeans=True)
        sns.histplot(x=x, bins=50, kde=True)

        ax_box.set(yticks=[])
        sns.despine(ax=ax_hist)
        sns.despine()

        plt.savefig(save_path + str(attribute))
        plt.close('all')
        print('')

# def scatter_plot(x,y):
