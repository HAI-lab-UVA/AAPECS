from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import os

result_path = '/Users/ry4jr/Library/Mobile Documents/com~apple~CloudDocs/AAPEX/results/heatmap'
file_name = 'feat_score_median_merge_comm.csv'
fig_name = 'feat_rank_corr_merge_comm.png'
os.chdir(result_path)
df_score_corr = pd.read_csv(file_name, index_col = 0)
sns.set_theme(style="white")
f, ax = plt.subplots(figsize=(34, 6))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(data = df_score_corr, vmin=-1, annot=True, cmap=cmap, mask = df_score_corr == 0, linewidths=.5, annot_kws={'size':7.4, 'weight':'bold'}, cbar_kws={"orientation": "horizontal"})
ax.xaxis.tick_top()
plt.xticks(rotation=90)
# plt.title('Feature Ranking')
#
plt.savefig(fig_name, bbox_inches='tight', dpi = 300)

