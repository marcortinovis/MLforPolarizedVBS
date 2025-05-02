import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import os

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # scalednorm feats data

    lep_data_tt = np.load(os.path.join(script_dir, '../processed_ntuples/vanilla/TT/scalednorm/lep_data_tt.npy'))
    vbsjet_data_tt = np.load(os.path.join(script_dir, '../processed_ntuples/vanilla/TT/scalednorm/vbsjet_data_tt.npy'))
    nvbsjet_data_tt = np.load(os.path.join(script_dir, '../processed_ntuples/vanilla/TT/scalednorm/nvbsjet_data_tt.npy'))
    jj_data_tt = np.load(os.path.join(script_dir, '../processed_ntuples/vanilla/TT/scalednorm/jj_data_tt.npy'))
    met_data_tt = np.load(os.path.join(script_dir, '../processed_ntuples/vanilla/TT/scalednorm/met_data_tt.npy'))
    nu_data_tt = np.load(os.path.join(script_dir, '../processed_ntuples/vanilla/TT/scalednorm/nu_data_tt.npy'))

    lep_data_tl = np.load(os.path.join(script_dir, '../processed_ntuples/vanilla/TL/scalednorm/lep_data_tl.npy'))
    vbsjet_data_tl = np.load(os.path.join(script_dir, '../processed_ntuples/vanilla/TL/scalednorm/vbsjet_data_tl.npy'))
    nvbsjet_data_tl = np.load(os.path.join(script_dir, '../processed_ntuples/vanilla/TL/scalednorm/nvbsjet_data_tl.npy'))
    jj_data_tl = np.load(os.path.join(script_dir, '../processed_ntuples/vanilla/TL/scalednorm/jj_data_tl.npy'))
    met_data_tl = np.load(os.path.join(script_dir, '../processed_ntuples/vanilla/TL/scalednorm/met_data_tl.npy'))
    nu_data_tl = np.load(os.path.join(script_dir, '../processed_ntuples/vanilla/TL/scalednorm/nu_data_tl.npy'))

    lep_data_ll = np.load(os.path.join(script_dir, '../processed_ntuples/vanilla/LL/scalednorm/lep_data_ll.npy'))
    vbsjet_data_ll = np.load(os.path.join(script_dir, '../processed_ntuples/vanilla/LL/scalednorm/vbsjet_data_ll.npy'))
    nvbsjet_data_ll = np.load(os.path.join(script_dir, '../processed_ntuples/vanilla/LL/scalednorm/nvbsjet_data_ll.npy'))
    jj_data_ll = np.load(os.path.join(script_dir, '../processed_ntuples/vanilla/LL/scalednorm/jj_data_ll.npy'))
    met_data_ll = np.load(os.path.join(script_dir, '../processed_ntuples/vanilla/LL/scalednorm/met_data_ll.npy'))
    nu_data_ll = np.load(os.path.join(script_dir, '../processed_ntuples/vanilla/LL/scalednorm/nu_data_ll.npy'))

    lep_dic_tt = {'lepPt1': lep_data_tt.T[0,0], 'lepPt2': lep_data_tt.T[0,1], 'lepEta1': lep_data_tt.T[1,0], 'lepEta2': lep_data_tt.T[1,1], 'lepPhi1': lep_data_tt.T[2,0], 'lepPhi2': lep_data_tt.T[2,1]}
    vbsjet_dic_tt = {'vbsjetPt1': vbsjet_data_tt.T[0,0], 'vbsjetPt2': vbsjet_data_tt.T[0,1], 'vbsjetEta1': vbsjet_data_tt.T[1,0], 'vbsjetEta2': vbsjet_data_tt.T[1,1], 'vbsjetPhi1': vbsjet_data_tt.T[2,0], 'vbsjetPhi2': vbsjet_data_tt.T[2,1], 'vbsjetM1': vbsjet_data_tt.T[3,0], 'vbsjetM2': vbsjet_data_tt.T[3,1]}
    nvbsjet_dic_tt = {'nvbsjetPt1': nvbsjet_data_tt.T[0,0], 'nvbsjetPt2': nvbsjet_data_tt.T[0,1], 'nvbsjetEta1': nvbsjet_data_tt.T[1,0], 'nvbsjetEta2': nvbsjet_data_tt.T[1,1], 'nvbsjetPhi1': nvbsjet_data_tt.T[2,0], 'nvbsjetPhi2': nvbsjet_data_tt.T[2,1], 'nvbsjetM1': nvbsjet_data_tt.T[3,0], 'nvbsjetM2': nvbsjet_data_tt.T[3,1]}
    jj_dic_tt = {'mjj': jj_data_tt.T[0,0], 'detajj': jj_data_tt.T[1,0]}
    met_dic_tt = {'ptMET': met_data_tt.T[0,0], 'phiMET': met_data_tt.T[1,0]}
    nu_dic_tt = {'ptv1': nu_data_tt.T[0,0], 'ptv2': nu_data_tt.T[0,1], 'etav1': nu_data_tt.T[1,0], 'etav2': nu_data_tt.T[1,1], 'phiv1': nu_data_tt.T[2,0], 'phiv2': nu_data_tt.T[2,1]}
    dic_tt = lep_dic_tt | vbsjet_dic_tt | nvbsjet_dic_tt | jj_dic_tt | met_dic_tt | nu_dic_tt
    df_tt = pd.DataFrame(dic_tt)

    lep_dic_tl = {'lepPt1': lep_data_tl.T[0,0], 'lepPt2': lep_data_tl.T[0,1], 'lepEta1': lep_data_tl.T[1,0], 'lepEta2': lep_data_tl.T[1,1], 'lepPhi1': lep_data_tl.T[2,0], 'lepPhi2': lep_data_tl.T[2,1]}
    vbsjet_dic_tl = {'vbsjetPt1': vbsjet_data_tl.T[0,0], 'vbsjetPt2': vbsjet_data_tl.T[0,1], 'vbsjetEta1': vbsjet_data_tl.T[1,0], 'vbsjetEta2': vbsjet_data_tl.T[1,1], 'vbsjetPhi1': vbsjet_data_tl.T[2,0], 'vbsjetPhi2': vbsjet_data_tl.T[2,1], 'vbsjetM1': vbsjet_data_tl.T[3,0], 'vbsjetM2': vbsjet_data_tl.T[3,1]}
    nvbsjet_dic_tl = {'nvbsjetPt1': nvbsjet_data_tl.T[0,0], 'nvbsjetPt2': nvbsjet_data_tl.T[0,1], 'nvbsjetEta1': nvbsjet_data_tl.T[1,0], 'nvbsjetEta2': nvbsjet_data_tl.T[1,1], 'nvbsjetPhi1': nvbsjet_data_tl.T[2,0], 'nvbsjetPhi2': nvbsjet_data_tl.T[2,1], 'nvbsjetM1': nvbsjet_data_tl.T[3,0], 'nvbsjetM2': nvbsjet_data_tl.T[3,1]}
    jj_dic_tl = {'mjj': jj_data_tl.T[0,0], 'detajj': jj_data_tl.T[1,0]}
    met_dic_tl = {'ptMET': met_data_tl.T[0,0], 'phiMET': met_data_tl.T[1,0]}
    nu_dic_tl = {'ptv1': nu_data_tl.T[0,0], 'ptv2': nu_data_tl.T[0,1], 'etav1': nu_data_tl.T[1,0], 'etav2': nu_data_tl.T[1,1], 'phiv1': nu_data_tl.T[2,0], 'phiv2': nu_data_tl.T[2,1]}
    dic_tl = lep_dic_tl | vbsjet_dic_tl | nvbsjet_dic_tl | jj_dic_tl | met_dic_tl | nu_dic_tl
    df_tl = pd.DataFrame(dic_tl)

    lep_dic_ll = {'lepPt1': lep_data_ll.T[0,0], 'lepPt2': lep_data_ll.T[0,1], 'lepEta1': lep_data_ll.T[1,0], 'lepEta2': lep_data_ll.T[1,1], 'lepPhi1': lep_data_ll.T[2,0], 'lepPhi2': lep_data_ll.T[2,1]}
    vbsjet_dic_ll = {'vbsjetPt1': vbsjet_data_ll.T[0,0], 'vbsjetPt2': vbsjet_data_ll.T[0,1], 'vbsjetEta1': vbsjet_data_ll.T[1,0], 'vbsjetEta2': vbsjet_data_ll.T[1,1], 'vbsjetPhi1': vbsjet_data_ll.T[2,0], 'vbsjetPhi2': vbsjet_data_ll.T[2,1], 'vbsjetM1': vbsjet_data_ll.T[3,0], 'vbsjetM2': vbsjet_data_ll.T[3,1]}
    nvbsjet_dic_ll = {'nvbsjetPt1': nvbsjet_data_ll.T[0,0], 'nvbsjetPt2': nvbsjet_data_ll.T[0,1], 'nvbsjetEta1': nvbsjet_data_ll.T[1,0], 'nvbsjetEta2': nvbsjet_data_ll.T[1,1], 'nvbsjetPhi1': nvbsjet_data_ll.T[2,0], 'nvbsjetPhi2': nvbsjet_data_ll.T[2,1], 'nvbsjetM1': nvbsjet_data_ll.T[3,0], 'nvbsjetM2': nvbsjet_data_ll.T[3,1]}
    jj_dic_ll = {'mjj': jj_data_ll.T[0,0], 'detajj': jj_data_ll.T[1,0]}
    met_dic_ll = {'ptMET': met_data_ll.T[0,0], 'phiMET': met_data_ll.T[1,0]}
    nu_dic_ll = {'ptv1': nu_data_ll.T[0,0], 'ptv2': nu_data_ll.T[0,1], 'etav1': nu_data_ll.T[1,0], 'etav2': nu_data_ll.T[1,1], 'phiv1': nu_data_ll.T[2,0], 'phiv2': nu_data_ll.T[2,1]}
    dic_ll = lep_dic_ll | vbsjet_dic_ll | nvbsjet_dic_ll | jj_dic_ll | met_dic_ll | nu_dic_ll
    df_ll = pd.DataFrame(dic_ll)

    data_features = ['lepPt1', 'lepPt2', 'lepEta1', 'lepEta2', 'lepPhi1', 'lepPhi2', 'vbsjetPt1', 'vbsjetPt2',
                     'vbsjetEta1', 'vbsjetEta2', 'vbsjetPhi1', 'vbsjetPhi2', 'vbsjetM1', 'vbsjetM2', 'nvbsjetPt1',
                     'nvbsjetPt2', 'nvbsjetEta1', 'nvbsjetEta2', 'nvbsjetPhi1', 'nvbsjetPhi2', 'nvbsjetM1', 'nvbsjetM2',
                     'mjj', 'detajj', 'ptMET', 'phiMET', 'ptv1', 'ptv2', 'etav1', 'etav2', 'phiv1', 'phiv2']

    plt.ioff()

    for feature in data_features:
        fig, ax = plt.subplots(ncols=1, nrows=2, gridspec_kw={'height_ratios': [3, 1]}, sharex='col') # figsize=(1.2*6.4, 1.4*4.8)
        bins_ll, edges_ll, hist_ll = ax[0].hist([x for x in df_ll[feature].to_numpy() if x != 0.], bins=50, density=True, histtype='step')
        bins_tl, edges_tl, hist_tl = ax[0].hist([x for x in df_tl[feature].to_numpy() if x != 0.], bins=50, density=True, histtype='step')
        bins_tt, edges_tt, hist_tt = ax[0].hist([x for x in df_tt[feature].to_numpy() if x != 0.], bins=50, density=True, histtype='step')
        ax[0].tick_params(labelbottom=True)
        ax[0].set_ylabel('Normalized counts')
        ax[0].tick_params(labelbottom=False, bottom=False)
        edges_tl = [0.5*(e1+e2) for e1, e2 in zip(edges_tl[:-1], edges_tl[1:])]
        edges_tt = [0.5*(e1+e2) for e1, e2 in zip(edges_ll[:-1], edges_ll[1:])]
        pulls_tl = [(a-b)/(b+1.e-34) for a, b in zip(bins_ll, bins_tl)]
        pulls_tt = [(a-b)/(b+1.e-34) for a, b in zip(bins_ll, bins_tt)]
        ax[1].scatter(edges_tt, pulls_tt, s=5, color='green')
        ax[1].scatter(edges_tl, pulls_tl, s=5, color='orange')
        xmin, xmax = ax[1].get_xlim()
        ax[1].hlines(0., xmin, xmax, color='k', alpha=0.5)
        ax[1].set_xlim(xmin, xmax)
        ax[1].set_ylabel('Residues')
        ax[1].set_xlabel(feature)
        ax[1].set_ylim(-2.5, 2.5)
        if max(np.concatenate((np.abs(pulls_tl), np.abs(pulls_tt)))) < 1.25:
            if max(np.concatenate((np.abs(pulls_tl), np.abs(pulls_tt)))) < 0.75:
                ax[1].set_ylim(-0.75, 0.75)
            else:
                ax[1].set_ylim(-1.25, 1.25)
        fig.subplots_adjust(hspace=0.05)
        fig.suptitle(f'Scaled and normalized {feature} histogram')
        fig.legend([hist_ll[0], hist_tl[0], hist_tt[0]], ['LL', 'TL', 'TT'], loc="upper right", bbox_to_anchor=(0.9,0.88))
        plt.savefig(os.path.join(script_dir, f'scalednorm/{feature}.png'), dpi=300.)
        print(f'scalednorm/{feature} done')


    # source feats data

    lep_data_tt = np.load(os.path.join(script_dir, '../processed_ntuples/vanilla/TT/source/lep_source_data_tt.npy'))
    vbsjet_data_tt = np.load(os.path.join(script_dir, '../processed_ntuples/vanilla/TT/source/vbsjet_source_data_tt.npy'))
    nvbsjet_data_tt = np.load(os.path.join(script_dir, '../processed_ntuples/vanilla/TT/source/nvbsjet_source_data_tt.npy'))
    jj_data_tt = np.load(os.path.join(script_dir, '../processed_ntuples/vanilla/TT/source/jj_source_data_tt.npy'))
    met_data_tt = np.load(os.path.join(script_dir, '../processed_ntuples/vanilla/TT/source/met_source_data_tt.npy'))
    nu_data_tt = np.load(os.path.join(script_dir, '../processed_ntuples/vanilla/TT/source/nu_source_data_tt.npy'))

    lep_data_tl = np.load(os.path.join(script_dir, '../processed_ntuples/vanilla/TL/source/lep_source_data_tl.npy'))
    vbsjet_data_tl = np.load(os.path.join(script_dir, '../processed_ntuples/vanilla/TL/source/vbsjet_source_data_tl.npy'))
    nvbsjet_data_tl = np.load(os.path.join(script_dir, '../processed_ntuples/vanilla/TL/source/nvbsjet_source_data_tl.npy'))
    jj_data_tl = np.load(os.path.join(script_dir, '../processed_ntuples/vanilla/TL/source/jj_source_data_tl.npy'))
    met_data_tl = np.load(os.path.join(script_dir, '../processed_ntuples/vanilla/TL/source/met_source_data_tl.npy'))
    nu_data_tl = np.load(os.path.join(script_dir, '../processed_ntuples/vanilla/TL/source/nu_source_data_tl.npy'))

    lep_data_ll = np.load(os.path.join(script_dir, '../processed_ntuples/vanilla/LL/source/lep_source_data_ll.npy'))
    vbsjet_data_ll = np.load(os.path.join(script_dir, '../processed_ntuples/vanilla/LL/source/vbsjet_source_data_ll.npy'))
    nvbsjet_data_ll = np.load(os.path.join(script_dir, '../processed_ntuples/vanilla/LL/source/nvbsjet_source_data_ll.npy'))
    jj_data_ll = np.load(os.path.join(script_dir, '../processed_ntuples/vanilla/LL/source/jj_source_data_ll.npy'))
    met_data_ll = np.load(os.path.join(script_dir, '../processed_ntuples/vanilla/LL/source/met_source_data_ll.npy'))
    nu_data_ll = np.load(os.path.join(script_dir, '../processed_ntuples/vanilla/LL/source/nu_source_data_ll.npy'))

    lep_dic_tt = {'lepPt1': lep_data_tt.T[0,0], 'lepPt2': lep_data_tt.T[0,1], 'lepEta1': lep_data_tt.T[1,0], 'lepEta2': lep_data_tt.T[1,1], 'lepPhi1': lep_data_tt.T[2,0], 'lepPhi2': lep_data_tt.T[2,1]}
    vbsjet_dic_tt = {'vbsjetPt1': vbsjet_data_tt.T[0,0], 'vbsjetPt2': vbsjet_data_tt.T[0,1], 'vbsjetEta1': vbsjet_data_tt.T[1,0], 'vbsjetEta2': vbsjet_data_tt.T[1,1], 'vbsjetPhi1': vbsjet_data_tt.T[2,0], 'vbsjetPhi2': vbsjet_data_tt.T[2,1], 'vbsjetM1': vbsjet_data_tt.T[3,0], 'vbsjetM2': vbsjet_data_tt.T[3,1]}
    nvbsjet_dic_tt = {'nvbsjetPt1': nvbsjet_data_tt.T[0,0], 'nvbsjetPt2': nvbsjet_data_tt.T[0,1], 'nvbsjetEta1': nvbsjet_data_tt.T[1,0], 'nvbsjetEta2': nvbsjet_data_tt.T[1,1], 'nvbsjetPhi1': nvbsjet_data_tt.T[2,0], 'nvbsjetPhi2': nvbsjet_data_tt.T[2,1], 'nvbsjetM1': nvbsjet_data_tt.T[3,0], 'nvbsjetM2': nvbsjet_data_tt.T[3,1]}
    jj_dic_tt = {'mjj': jj_data_tt.T[0,0], 'detajj': jj_data_tt.T[1,0]}
    met_dic_tt = {'ptMET': met_data_tt.T[0,0], 'phiMET': met_data_tt.T[1,0]}
    nu_dic_tt = {'ptv1': nu_data_tt.T[0,0], 'ptv2': nu_data_tt.T[0,1], 'etav1': nu_data_tt.T[1,0], 'etav2': nu_data_tt.T[1,1], 'phiv1': nu_data_tt.T[2,0], 'phiv2': nu_data_tt.T[2,1]}
    dic_tt = lep_dic_tt | vbsjet_dic_tt | nvbsjet_dic_tt | jj_dic_tt | met_dic_tt | nu_dic_tt
    df_tt = pd.DataFrame(dic_tt)

    lep_dic_tl = {'lepPt1': lep_data_tl.T[0,0], 'lepPt2': lep_data_tl.T[0,1], 'lepEta1': lep_data_tl.T[1,0], 'lepEta2': lep_data_tl.T[1,1], 'lepPhi1': lep_data_tl.T[2,0], 'lepPhi2': lep_data_tl.T[2,1]}
    vbsjet_dic_tl = {'vbsjetPt1': vbsjet_data_tl.T[0,0], 'vbsjetPt2': vbsjet_data_tl.T[0,1], 'vbsjetEta1': vbsjet_data_tl.T[1,0], 'vbsjetEta2': vbsjet_data_tl.T[1,1], 'vbsjetPhi1': vbsjet_data_tl.T[2,0], 'vbsjetPhi2': vbsjet_data_tl.T[2,1], 'vbsjetM1': vbsjet_data_tl.T[3,0], 'vbsjetM2': vbsjet_data_tl.T[3,1]}
    nvbsjet_dic_tl = {'nvbsjetPt1': nvbsjet_data_tl.T[0,0], 'nvbsjetPt2': nvbsjet_data_tl.T[0,1], 'nvbsjetEta1': nvbsjet_data_tl.T[1,0], 'nvbsjetEta2': nvbsjet_data_tl.T[1,1], 'nvbsjetPhi1': nvbsjet_data_tl.T[2,0], 'nvbsjetPhi2': nvbsjet_data_tl.T[2,1], 'nvbsjetM1': nvbsjet_data_tl.T[3,0], 'nvbsjetM2': nvbsjet_data_tl.T[3,1]}
    jj_dic_tl = {'mjj': jj_data_tl.T[0,0], 'detajj': jj_data_tl.T[1,0]}
    met_dic_tl = {'ptMET': met_data_tl.T[0,0], 'phiMET': met_data_tl.T[1,0]}
    nu_dic_tl = {'ptv1': nu_data_tl.T[0,0], 'ptv2': nu_data_tl.T[0,1], 'etav1': nu_data_tl.T[1,0], 'etav2': nu_data_tl.T[1,1], 'phiv1': nu_data_tl.T[2,0], 'phiv2': nu_data_tl.T[2,1]}
    dic_tl = lep_dic_tl | vbsjet_dic_tl | nvbsjet_dic_tl | jj_dic_tl | met_dic_tl | nu_dic_tl
    df_tl = pd.DataFrame(dic_tl)

    lep_dic_ll = {'lepPt1': lep_data_ll.T[0,0], 'lepPt2': lep_data_ll.T[0,1], 'lepEta1': lep_data_ll.T[1,0], 'lepEta2': lep_data_ll.T[1,1], 'lepPhi1': lep_data_ll.T[2,0], 'lepPhi2': lep_data_ll.T[2,1]}
    vbsjet_dic_ll = {'vbsjetPt1': vbsjet_data_ll.T[0,0], 'vbsjetPt2': vbsjet_data_ll.T[0,1], 'vbsjetEta1': vbsjet_data_ll.T[1,0], 'vbsjetEta2': vbsjet_data_ll.T[1,1], 'vbsjetPhi1': vbsjet_data_ll.T[2,0], 'vbsjetPhi2': vbsjet_data_ll.T[2,1], 'vbsjetM1': vbsjet_data_ll.T[3,0], 'vbsjetM2': vbsjet_data_ll.T[3,1]}
    nvbsjet_dic_ll = {'nvbsjetPt1': nvbsjet_data_ll.T[0,0], 'nvbsjetPt2': nvbsjet_data_ll.T[0,1], 'nvbsjetEta1': nvbsjet_data_ll.T[1,0], 'nvbsjetEta2': nvbsjet_data_ll.T[1,1], 'nvbsjetPhi1': nvbsjet_data_ll.T[2,0], 'nvbsjetPhi2': nvbsjet_data_ll.T[2,1], 'nvbsjetM1': nvbsjet_data_ll.T[3,0], 'nvbsjetM2': nvbsjet_data_ll.T[3,1]}
    jj_dic_ll = {'mjj': jj_data_ll.T[0,0], 'detajj': jj_data_ll.T[1,0]}
    met_dic_ll = {'ptMET': met_data_ll.T[0,0], 'phiMET': met_data_ll.T[1,0]}
    nu_dic_ll = {'ptv1': nu_data_ll.T[0,0], 'ptv2': nu_data_ll.T[0,1], 'etav1': nu_data_ll.T[1,0], 'etav2': nu_data_ll.T[1,1], 'phiv1': nu_data_ll.T[2,0], 'phiv2': nu_data_ll.T[2,1]}
    dic_ll = lep_dic_ll | vbsjet_dic_ll | nvbsjet_dic_ll | jj_dic_ll | met_dic_ll | nu_dic_ll
    df_ll = pd.DataFrame(dic_ll)

    data_features = ['lepPt1', 'lepPt2', 'lepEta1', 'lepEta2', 'lepPhi1', 'lepPhi2', 'vbsjetPt1', 'vbsjetPt2',
                     'vbsjetEta1', 'vbsjetEta2', 'vbsjetPhi1', 'vbsjetPhi2', 'vbsjetM1', 'vbsjetM2', 'nvbsjetPt1',
                     'nvbsjetPt2', 'nvbsjetEta1', 'nvbsjetEta2', 'nvbsjetPhi1', 'nvbsjetPhi2', 'nvbsjetM1', 'nvbsjetM2',
                     'mjj', 'detajj', 'ptMET', 'phiMET', 'ptv1', 'ptv2', 'etav1', 'etav2', 'phiv1', 'phiv2']

    plt.ioff()

    for feature in data_features:
        fig, ax = plt.subplots(ncols=1, nrows=2, gridspec_kw={'height_ratios': [3, 1]}, sharex='col') # figsize=(1.2*6.4, 1.4*4.8)
        bins_ll, edges_ll, hist_ll = ax[0].hist([x for x in df_ll[feature].to_numpy() if x != 0.], bins=50, density=True, histtype='step')
        bins_tl, edges_tl, hist_tl = ax[0].hist([x for x in df_tl[feature].to_numpy() if x != 0.], bins=50, density=True, histtype='step')
        bins_tt, edges_tt, hist_tt = ax[0].hist([x for x in df_tt[feature].to_numpy() if x != 0.], bins=50, density=True, histtype='step')
        ax[0].tick_params(labelbottom=True)
        ax[0].set_ylabel('Normalized counts')
        ax[0].tick_params(labelbottom=False, bottom=False)
        edges_tl = [0.5*(e1+e2) for e1, e2 in zip(edges_tl[:-1], edges_tl[1:])]
        edges_tt = [0.5*(e1+e2) for e1, e2 in zip(edges_ll[:-1], edges_ll[1:])]
        pulls_tl = [(a-b)/(b+1.e-34) for a, b in zip(bins_ll, bins_tl)]
        pulls_tt = [(a-b)/(b+1.e-34) for a, b in zip(bins_ll, bins_tt)]
        ax[1].scatter(edges_tt, pulls_tt, s=5, color='green')
        ax[1].scatter(edges_tl, pulls_tl, s=5, color='orange')
        xmin, xmax = ax[1].get_xlim()
        ax[1].hlines(0., xmin, xmax, color='k', alpha=0.5)
        ax[1].set_xlim(xmin, xmax)
        ax[1].set_ylabel('Residues')
        ax[1].set_xlabel(feature)
        ax[1].set_ylim(-2.5, 2.5)
        if max(np.concatenate((np.abs(pulls_tl), np.abs(pulls_tt)))) < 1.25:
            if max(np.concatenate((np.abs(pulls_tl), np.abs(pulls_tt)))) < 0.75:
                ax[1].set_ylim(-0.75, 0.75)
            else:
                ax[1].set_ylim(-1.25, 1.25)
        fig.subplots_adjust(hspace=0.05)
        fig.suptitle(f'Source {feature} histogram')
        fig.legend([hist_ll[0], hist_tl[0], hist_tt[0]], ['LL', 'TL', 'TT'], loc="upper right", bbox_to_anchor=(0.9,0.88))
        plt.savefig(os.path.join(script_dir, f'source/source_{feature}.png'))
        print(f'source/{feature} done')

    # scalednorm newvars

    dic_ll = {'costheta_1': np.load(os.path.join(script_dir, '../processed_ntuples/singlewrf_reconstruction/LL/costheta_1_nr_norm_ll.npy')),
            'costheta_2': np.load(os.path.join(script_dir, '../processed_ntuples/singlewrf_reconstruction/LL/costheta_2_nr_norm_ll.npy')),
            'cos_cs': np.load(os.path.join(script_dir, '../processed_ntuples/collinssoper/costhetas_ll.npy')),
            'cos_th_1': np.load(os.path.join(script_dir, '../processed_ntuples/transverse_helicity/LL/cos_th_1_nr_ll.npy')),
            'cos_th_2': np.load(os.path.join(script_dir, '../processed_ntuples/transverse_helicity/LL/cos_th_2_nr_ll.npy')),
            'r_pt': np.load(os.path.join(script_dir, '../processed_ntuples/ratio_pt/Rpt_ll.npy'))}
    df_ll = pd.DataFrame(dic_ll)

    dic_tl = {'costheta_1': np.load(os.path.join(script_dir, '../processed_ntuples/singlewrf_reconstruction/TL/costheta_1_nr_norm_tl.npy')),
            'costheta_2': np.load(os.path.join(script_dir, '../processed_ntuples/singlewrf_reconstruction/TL/costheta_2_nr_norm_tl.npy')),
            'cos_cs': np.load(os.path.join(script_dir, '../processed_ntuples/collinssoper/costhetas_tl.npy')),
            'cos_th_1': np.load(os.path.join(script_dir, '../processed_ntuples/transverse_helicity/TL/cos_th_1_nr_tl.npy')),
            'cos_th_2': np.load(os.path.join(script_dir, '../processed_ntuples/transverse_helicity/TL/cos_th_2_nr_tl.npy')),
            'r_pt': np.load(os.path.join(script_dir, '../processed_ntuples/ratio_pt/Rpt_tl.npy'))}
    df_tl = pd.DataFrame(dic_tl)

    dic_tt = {'costheta_1': np.load(os.path.join(script_dir, '../processed_ntuples/singlewrf_reconstruction/TT/costheta_1_nr_norm_tt.npy')),
            'costheta_2': np.load(os.path.join(script_dir, '../processed_ntuples/singlewrf_reconstruction/TT/costheta_2_nr_norm_tt.npy')),
            'cos_cs': np.load(os.path.join(script_dir, '../processed_ntuples/collinssoper/costhetas_tt.npy')),
            'cos_th_1': np.load(os.path.join(script_dir, '../processed_ntuples/transverse_helicity/TT/cos_th_1_nr_tt.npy')),
            'cos_th_2': np.load(os.path.join(script_dir, '../processed_ntuples/transverse_helicity/TT/cos_th_2_nr_tt.npy')),
            'r_pt': np.load(os.path.join(script_dir, '../processed_ntuples/ratio_pt/Rpt_tt.npy'))}
    df_tt = pd.DataFrame(dic_tt)

    newvar_features = ['costheta_1', 'costheta_2', 'cos_cs', 'cos_th_1', 'cos_th_2', 'r_pt']

    for feature in newvar_features:
        fig, ax = plt.subplots(ncols=1, nrows=2, gridspec_kw={'height_ratios': [3, 1]}, sharex='col') # figsize=(1.2*6.4, 1.4*4.8)
        bins_ll, edges_ll, hist_ll = ax[0].hist([x for x in df_ll[feature].to_numpy() if x != 0.], bins=50, density=True, histtype='step')
        bins_tl, edges_tl, hist_tl = ax[0].hist([x for x in df_tl[feature].to_numpy() if x != 0.], bins=50, density=True, histtype='step')
        bins_tt, edges_tt, hist_tt = ax[0].hist([x for x in df_tt[feature].to_numpy() if x != 0.], bins=50, density=True, histtype='step')
        ax[0].tick_params(labelbottom=True)
        ax[0].set_ylabel('Normalized counts')
        ax[0].tick_params(labelbottom=False, bottom=False)
        edges_tl = [0.5*(e1+e2) for e1, e2 in zip(edges_tl[:-1], edges_tl[1:])]
        edges_tt = [0.5*(e1+e2) for e1, e2 in zip(edges_ll[:-1], edges_ll[1:])]
        pulls_tl = [(a-b)/(b+1.e-34) for a, b in zip(bins_ll, bins_tl)]
        pulls_tt = [(a-b)/(b+1.e-34) for a, b in zip(bins_ll, bins_tt)]
        ax[1].scatter(edges_tt, pulls_tt, s=5, color='green')
        ax[1].scatter(edges_tl, pulls_tl, s=5, color='orange')
        xmin, xmax = ax[1].get_xlim()
        ax[1].hlines(0., xmin, xmax, color='k', alpha=0.5)
        ax[1].set_xlim(xmin, xmax)
        ax[1].set_ylabel('Residues')
        ax[1].set_xlabel(feature)
        ax[1].set_ylim(-2.5, 2.5)
        if max(np.concatenate((np.abs(pulls_tl), np.abs(pulls_tt)))) < 1.25:
            if max(np.concatenate((np.abs(pulls_tl), np.abs(pulls_tt)))) < 0.75:
                ax[1].set_ylim(-0.75, 0.75)
            else:
                ax[1].set_ylim(-1.25, 1.25)
        fig.subplots_adjust(hspace=0.05)
        fig.suptitle(f'Scaled and normalized {feature} histogram')
        fig.legend([hist_ll[0], hist_tl[0], hist_tt[0]], ['LL', 'TL', 'TT'], loc="upper right", bbox_to_anchor=(0.9,0.88))
        plt.savefig(os.path.join(script_dir, f'scalednorm/{feature}.png'))
        print(f'scalednorm/{feature} done')

    # source newvars

    dic_ll = {'costheta_1': np.load(os.path.join(script_dir, '../processed_ntuples/singlewrf_reconstruction/LL/costheta_1_nr_source_ll.npy')),
            'costheta_2': np.load(os.path.join(script_dir, '../processed_ntuples/singlewrf_reconstruction/LL/costheta_2_nr_source_ll.npy')),
            'cos_cs': np.load(os.path.join(script_dir, '../processed_ntuples/collinssoper/costhetas_source_ll.npy')),
            'cos_th_1': np.load(os.path.join(script_dir, '../processed_ntuples/transverse_helicity/LL/cos_th_1_nr_source_ll.npy')),
            'cos_th_2': np.load(os.path.join(script_dir, '../processed_ntuples/transverse_helicity/LL/cos_th_2_nr_source_ll.npy')),
            'r_pt': np.load(os.path.join(script_dir, '../processed_ntuples/ratio_pt/Rpt_source_ll.npy'))}
    df_ll = pd.DataFrame(dic_ll)

    dic_tl = {'costheta_1': np.load(os.path.join(script_dir, '../processed_ntuples/singlewrf_reconstruction/TL/costheta_1_nr_source_tl.npy')),
            'costheta_2': np.load(os.path.join(script_dir, '../processed_ntuples/singlewrf_reconstruction/TL/costheta_2_nr_source_tl.npy')),
            'cos_cs': np.load(os.path.join(script_dir, '../processed_ntuples/collinssoper/costhetas_source_tl.npy')),
            'cos_th_1': np.load(os.path.join(script_dir, '../processed_ntuples/transverse_helicity/TL/cos_th_1_nr_source_tl.npy')),
            'cos_th_2': np.load(os.path.join(script_dir, '../processed_ntuples/transverse_helicity/TL/cos_th_2_nr_source_tl.npy')),
            'r_pt': np.load(os.path.join(script_dir, '../processed_ntuples/ratio_pt/Rpt_source_tl.npy'))}
    df_tl = pd.DataFrame(dic_tl)

    dic_tt = {'costheta_1': np.load(os.path.join(script_dir, '../processed_ntuples/singlewrf_reconstruction/TT/costheta_1_nr_source_tt.npy')),
            'costheta_2': np.load(os.path.join(script_dir, '../processed_ntuples/singlewrf_reconstruction/TT/costheta_2_nr_source_tt.npy')),
            'cos_cs': np.load(os.path.join(script_dir, '../processed_ntuples/collinssoper/costhetas_source_tt.npy')),
            'cos_th_1': np.load(os.path.join(script_dir, '../processed_ntuples/transverse_helicity/TT/cos_th_1_nr_source_tt.npy')),
            'cos_th_2': np.load(os.path.join(script_dir, '../processed_ntuples/transverse_helicity/TT/cos_th_2_nr_source_tt.npy')),
            'r_pt': np.load(os.path.join(script_dir, '../processed_ntuples/ratio_pt/Rpt_source_tt.npy'))}
    df_tt = pd.DataFrame(dic_tt)

    newvar_features = ['costheta_1', 'costheta_2', 'cos_cs', 'cos_th_1', 'cos_th_2', 'r_pt']

    for feature in newvar_features:
        fig, ax = plt.subplots(ncols=1, nrows=2, gridspec_kw={'height_ratios': [3, 1]}, sharex='col') # figsize=(1.2*6.4, 1.4*4.8)
        bins_ll, edges_ll, hist_ll = ax[0].hist([x for x in df_ll[feature].to_numpy() if x != 0.], bins=50, density=True, histtype='step')
        bins_tl, edges_tl, hist_tl = ax[0].hist([x for x in df_tl[feature].to_numpy() if x != 0.], bins=50, density=True, histtype='step')
        bins_tt, edges_tt, hist_tt = ax[0].hist([x for x in df_tt[feature].to_numpy() if x != 0.], bins=50, density=True, histtype='step')
        ax[0].tick_params(labelbottom=True)
        ax[0].set_ylabel('Normalized counts')
        ax[0].tick_params(labelbottom=False, bottom=False)
        edges_tl = [0.5*(e1+e2) for e1, e2 in zip(edges_tl[:-1], edges_tl[1:])]
        edges_tt = [0.5*(e1+e2) for e1, e2 in zip(edges_ll[:-1], edges_ll[1:])]
        pulls_tl = [(a-b)/(b+1.e-34) for a, b in zip(bins_ll, bins_tl)]
        pulls_tt = [(a-b)/(b+1.e-34) for a, b in zip(bins_ll, bins_tt)]
        ax[1].scatter(edges_tt, pulls_tt, s=5, color='green')
        ax[1].scatter(edges_tl, pulls_tl, s=5, color='orange')
        xmin, xmax = ax[1].get_xlim()
        ax[1].hlines(0., xmin, xmax, color='k', alpha=0.5)
        ax[1].set_xlim(xmin, xmax)
        ax[1].set_ylabel('Residues')
        ax[1].set_xlabel(feature)
        ax[1].set_ylim(-2.5, 2.5)
        if max(np.concatenate((np.abs(pulls_tl), np.abs(pulls_tt)))) < 1.25:
            if max(np.concatenate((np.abs(pulls_tl), np.abs(pulls_tt)))) < 0.75:
                ax[1].set_ylim(-0.75, 0.75)
            else:
                ax[1].set_ylim(-1.25, 1.25)
        fig.subplots_adjust(hspace=0.05)
        fig.suptitle(f'Source {feature} histogram')
        fig.legend([hist_ll[0], hist_tl[0], hist_tt[0]], ['LL', 'TL', 'TT'], loc="upper right", bbox_to_anchor=(0.9,0.88))
        plt.savefig(os.path.join(script_dir, f'source/source_{feature}.png'))
        print(f'source/{feature} done')

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 

if __name__ == "__main__":
	main()