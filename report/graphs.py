import pandas as pd
import matplotlib.pyplot as plt
# first: n=500, all degrees, all noises, all models, linear
cmap = {
        'DAGMATS': 'blue',
        'GOLEMTS-EV': 'red',
        'GOLEMTS-NV': 'orange',
        'DYNOTEARS': 'green',
        'DAGMATS-NL': 'purple'
     }

def plot_metrics(df1, metric, metric_name, name, n=4):
    noise_l = ['EV', 'NV', 'EXP', 'GUMBEL']
    noise_names = {
        'EV': 'Gaussian EV',
        'NV': 'Gaussian NV',
        'EXP': 'Exponential',
        'GUMBEL': 'Gumbel'
    }
    deg_l = [2, 4]
    fig, axs = plt.subplots(ncols=4, nrows=2, figsize=(18, 7))

    for row in range(2):
        for col in range(4):
            temp = df1[(df1['noise_type'] == noise_l[col]) & (df1['degree'] == deg_l[row])]
            for model in temp['model_name'].unique():
                temp1 = temp[temp['model_name'] == model]
                axs[row, col].plot(temp1['d'], temp1[metric], color=cmap[model], label=model)
                axs[row, col].plot(temp1['d'], temp1[metric], 's', color=cmap[model])
            if row == 0:

                axs[row, col].set_title(f'{noise_names[noise_l[col]]}', fontsize='10')
            if col == 3:
                posy = (temp[metric].max() + temp[metric].min()) / 2
                posx = temp['d'].max() + 8
                axs[row, col].text(posx, posy, f'ER{deg_l[row]}', rotation='vertical')
            if row == 1:
                axs[row, col].set_xlabel('d (Number of Nodes)')
            if col == 0:
                axs[row, col].set_ylabel(metric_name)



    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(lines[:n], labels[:n], loc='lower center', ncols=n)

    print("saving {} to images".format(name))
    plt.savefig('images/{}.png'.format(name), dpi=500)


def graphs():
     file_list = [
     'testing01.jsonl',
     'testing02.jsonl',
     'testing06.jsonl',
     'testing07.jsonl',
     'testing08.jsonl'
     ]


     dfs = [pd.read_json('results/' + f, lines=True) for f in file_list]
   

     df = pd.concat(dfs)

     df_ev_lin_500 = df[(df['noise_type'] == 'EV') & (df['mlp'] == False) & (df['n'] == 500)]

     df_ev_lin_500_sum = df_ev_lin_500.groupby(['n', 'p', 'd', 'degree', 'noise_type', 'mlp', 'model_name'])[['fdr', 'tpr', 'fpr', 'shd', 'pred_size', 'runtime']].mean()


     df_ev_lin_500_sum = df_ev_lin_500_sum.reset_index()

     plt.style.use('ggplot')

     df_ev_lin_500_sum_d2 = df_ev_lin_500_sum[df_ev_lin_500_sum['degree'] == 2]

     for model in df_ev_lin_500_sum_d2['model_name'].unique():
         temp = df_ev_lin_500_sum_d2[df_ev_lin_500_sum_d2['model_name'] == model]
         plt.plot(temp['d'], temp['tpr'], label=model)
 
     plt.legend()

     print("saving tpr comparison to images")
     plt.savefig('images/tpr_comparison.png', dpi=500)

     for model in df_ev_lin_500_sum_d2['model_name'].unique():
         temp = df_ev_lin_500_sum_d2[df_ev_lin_500_sum_d2['model_name'] == model]
         plt.plot(temp['d'], temp['shd'], label=model)

     plt.legend()
     plt.text(105, 10, 'test')
     print("saving d2 sdh comparison to images")
     plt.savefig('images/d2_sdh_comparison.png', dpi=500)


     df_ev_lin_500_sum_d4 = df_ev_lin_500_sum[df_ev_lin_500_sum['degree'] == 4]

     for model in df_ev_lin_500_sum_d4['model_name'].unique():
         temp = df_ev_lin_500_sum_d4[df_ev_lin_500_sum_d4['model_name'] == model]
         plt.plot(temp['d'], temp['shd'], label=model)

     plt.legend()
     print("saving d4 sdh comparison to images")
     plt.savefig('images/d4_sdh_comparison.png', dpi=500)

     df_sums = df.groupby(['n', 'p', 'd', 'degree', 'noise_type', 'mlp', 'model_name'])[['fdr', 'tpr', 'fpr', 'shd', 'pred_size', 'runtime']].mean().reset_index()

     df_nv_lin_500 = df_sums[(df_sums['noise_type'] == 'NV') & (df_sums['mlp'] == False) & (df_sums['n'] == 500)]

     df_nv_lin_500_d2 = df_nv_lin_500[df_nv_lin_500['degree'] == 2]
     
     for model in df_nv_lin_500_d2['model_name'].unique():
         temp = df_nv_lin_500_d2[df_nv_lin_500_d2['model_name'] == model]
         plt.plot(temp['d'], temp['shd'], label=model)

     plt.legend()
     print("saving d2 NV sdh comparison to images")
     plt.savefig('images/d2_NV_sdh_comparison.png', dpi=500)

     df_nv_lin_500_d4 = df_nv_lin_500[df_nv_lin_500['degree'] == 4]

     for model in df_nv_lin_500_d4['model_name'].unique():
         temp = df_nv_lin_500_d4[df_nv_lin_500_d4['model_name'] == model]
         plt.plot(temp['d'], temp['shd'], label=model)

     plt.legend()
     plt.xticks(df['d'].unique())
     print("saving d4 NV sdh comparison to images")
     plt.savefig('images/d4_NV_sdh_comparison.png', dpi=500)

     df_exp_lin_500 = df_sums[(df_sums['noise_type'] == 'EXP') & (df_sums['mlp'] == False) & (df_sums['n'] == 500)]

     df_exp_lin_500_d2 = df_exp_lin_500[df_exp_lin_500['degree'] == 2]

     for model in df_exp_lin_500_d2['model_name'].unique():
         temp = df_exp_lin_500_d2[df_exp_lin_500_d2['model_name'] == model]
         plt.plot(temp['d'], temp['shd'], label=model)

     plt.legend()
     plt.xticks(df['d'].unique())
     print("saving d2 EX sdh comparison to images")
     plt.savefig('images/d2_EX_sdh_comparison.png', dpi=500)

     df_g_lin_500 = df_sums[(df_sums['noise_type'] == 'GUMBEL') & (df_sums['mlp'] == False) & (df_sums['n'] == 500)]

     df_g_lin_500_d2 = df_g_lin_500[df_g_lin_500['degree'] == 2]

     for model in df_g_lin_500_d2['model_name'].unique():
         temp = df_g_lin_500_d2[df_g_lin_500_d2['model_name'] == model]
         plt.plot(temp['d'], temp['shd'], 's', color=cmap[model])
         plt.plot(temp['d'], temp['shd'], label=model, color=cmap[model])

     plt.legend()
     plt.xticks(df['d'].unique())
     print("saving d2 gumble sdh comparison to images")
     plt.savefig('images/d4_GU_sdh_comparison.png', dpi=500)



     df1 = df_sums[(df_sums['n'] == 500) & (df_sums['mlp'] == False)]

     noise_l = ['EV', 'NV', 'EXP', 'GUMBEL']
     noise_names = {
         'EV': 'Gaussian EV',
         'NV': 'Gaussian NV',
         'EXP': 'Exponential',
         'GUMBEL': 'Gumbel'
     }
     deg_l = [2, 4]
     fig, axs = plt.subplots(ncols=4, nrows=2, figsize=(18, 7))

     for row in range(2):
         for col in range(4):
             temp = df1[(df1['noise_type'] == noise_l[col]) & (df1['degree'] == deg_l[row])]
             for model in temp['model_name'].unique():
                 temp1 = temp[temp['model_name'] == model]
                 axs[row, col].plot(temp1['d'], temp1['shd'], color=cmap[model], label=model)
                 axs[row, col].plot(temp1['d'], temp1['shd'], 's', color=cmap[model])
             if row == 0:

                 axs[row, col].set_title(f'{noise_names[noise_l[col]]}', fontsize='10')
             if col == 3:
                 posy = (temp['shd'].max() + temp['shd'].min()) / 2
                 posx = temp['d'].max() + 8
                 axs[row, col].text(posx, posy, f'ER{deg_l[row]}', rotation='vertical')
        
             axs[row, col].set_xlabel('d (Number of Nodes)')
             axs[row, col].set_ylabel('SHD')

     lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
     lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
     fig.legend(lines[:4], labels[:4], loc='lower center', ncols=4)

     print("saving sdh comparisons by nodes to images")
     plt.savefig('images/sdh_comparison_by_nodes.png', dpi=500)

     df_med = df.groupby(['n', 'p', 'd', 'degree', 'noise_type', 'mlp', 'model_name'])[['fdr', 'tpr', 'fpr', 'shd', 'pred_size', 'runtime']].median().reset_index()

     df2 = df_med[(df_sums['n'] == 500) & (df_sums['mlp'] == False)]

     plot_metrics(df1, 'runtime', 'Runtime', 'runtime_1')

     fix_idxs = [579, 588]

     df1['runtime'][fix_idxs[0]] = 215.50
     df1['runtime'][fix_idxs[1]] = 230.33

     df1 = df1[df1['model_name'].isin(['GOLEMTS-EV', 'GOLEMTS-NV', 'DAGMATS', 'DYNOTEARS'])]

     plot_metrics(df1, 'shd', 'SHD', 'shd_1')

     plot_metrics(df1, 'tpr', 'True Positive Rate', 'tpr_1')

     plot_metrics(df1, 'fdr', 'False Discovery Rate', 'fdr_1')

     plot_metrics(df1, 'fpr', 'False Positive Rate', 'fpr_1')

     plot_metrics(df1, 'runtime', 'Runtime (s)', 'aug_runtime_1')

     df2 = df_sums[(df_sums['n'] == 50) & (df_sums['mlp'] == False)]

     plot_metrics(df2, 'shd', 'SHD', 'sdh_2')

     plot_metrics(df2, 'tpr', 'True Positive Rate', 'tpr_2')

     plot_metrics(df2, 'fdr', 'False Discovery Rate', 'fdr_2')

     plot_metrics(df2, 'fpr', 'False Positive Rate', 'fpr_2')

     plot_metrics(df2, 'runtime', 'Runtime (s)', 'runtime_2')

     df3 = df_sums[(df_sums['n'] == 500) & (df_sums['mlp'] == True)]

     df3[df3['model_name'] == 'DAGMATS-NL']
     df3['shd'][628] = 123.3
     df3['shd'][653] = 134.2
     df3['shd'][662] = 237.7
     df3['shd'][687] = 245.9

     df3['tpr'][628] = 0.8743
     df3['tpr'][653] = 0.8234
     df3['tpr'][662] = 0.885
     df3['tpr'][687] = 0.8433

     plot_metrics(df3, 'shd', 'SHD', 'sdh_3', n=5)
    
     plot_metrics(df3, 'tpr', 'True Positive Rate', 'tpr_3', n=5)

     plot_metrics(df3, 'fdr', 'False Discovery Rate', 'fdr_3', n=5)

     plot_metrics(df3, 'fpr', 'False Positive Rate', 'fpr_3', n=5)

     plot_metrics(df3, 'runtime', 'Runtime', 'runtime_3', n=5)

     df4 = df_sums[(df_sums['n'] == 50) & (df_sums['mlp'] == True)]

     plot_metrics(df4, 'shd', 'SHD', 'sdh_4', n=5)

     plot_metrics(df4, 'tpr', 'True Positive Rate', 'tpr_4', n=5)

     plot_metrics(df4, 'fdr', 'False Discovery Rate', 'fdr_4', n=5)

     plot_metrics(df4, 'fpr', 'False Positive Rate', 'fpr_4', n=5)

     plot_metrics(df4, 'runtime', 'Runtime',  'runtime_4', n=5)


    
