import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

dataset = pd.read_csv("color_features.csv")

# sns.boxplot(x='varieties', y='mean_v', data=dataset)

batanes = dataset[dataset['varieties']==0]['mean_s']
ilocos_pink = dataset[dataset['varieties']==1]['mean_s']
ilocos_white = dataset[dataset['varieties']==2]['mean_s']
mexican = dataset[dataset['varieties']==3]['mean_s']
mmsu_gem = dataset[dataset['varieties']==4]['mean_s']
tanbolters = dataset[dataset['varieties']==5]['mean_s']
vfta = dataset[dataset['varieties']==6]['mean_s']

fig = plt.figure(1)
fig.suptitle('Mean Saturation', fontsize=14, fontweight='bold')
plt.xlabel("Different Garlic Varieties")
plt.ylabel("Pixel Intensity")
plt.boxplot([batanes, ilocos_pink, ilocos_white, mexican, mmsu_gem, tanbolters, vfta], 
	labels=['batanes','ilocos_pink','ilocos_white','mexican', 'mmsu_gem', 'tanbolters', 'vfta'], showfliers=False)
# fig.savefig('Mean_R.png',dpi=100)
plt.show()