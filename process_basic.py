import pandas as pd
import numpy as np
import argparse
from scipy.stats import multinomial, chisquare
from tqdm import tqdm

translate = {'TTT':'F','TTC':'F','TTA':'L','TTG':'L','TCT':'S','TCC':'S','TCA':'S','TCG':'S',
            'TAT':'Y','TAC':'Y','TAA':'Stop','TAG':'Stop','TGT':'C','TGC':'C','TGA':'Stop','TGG':'W',
            'CTT':'L','CTC':'L','CTA':'L','CTG':'L','CCT':'P','CCC':'P','CCA':'P','CCG':'P',
            'CAT':'H','CAC':'H','CAA':'Q','CAG':'Q','CGT':'R','CGC':'R','CGA':'R','CGG':'R',
        'ATT':'I','ATC':'I','ATA':'I','ATG':'M','ACT':'T','ACC':'T','ACA':'T','ACG':'T',
            'AAT':'N','AAC':'N','AAA':'K','AAG':'K','AGT':'S','AGC':'S','AGA':'R','AGG':'R',
            'GTT':'V','GTC':'V','GTA':'V','GTG':'V','GCT':'A','GCC':'A','GCA':'A','GCG':'A',
            'GAT':'D','GAC':'D','GAA':'E','GAG':'E','GGT':'G','GGC':'G','GGA':'G','GGG':'G'}
aagroups = {'aliphatic':['G','A','V','L','I'], 'hydroxyl':['S','C','U','T','M'], 'cyclic':['P'], 'aromatic':['F','Y','W'], 'basic':['H','K','R'], 'acidic':['D','E','N','Q'], 'N/A':['Stop']}   


def parse_basic_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--translation", help="Path to a translation table produced by matUtils extract.", required=True)
    parser.add_argument("-g", "--gtf", help="Path to a gtf containing exon spans", default='exons.gtf')
    parser.add_argument("-f", "--fasta", help="Path to a fasta containing the gene sequences matching the exons", default='genes.fasta')
    parser.add_argument("-c", "--clades", help="Path to a file containing node and clade matching annotations.", default='all_nodes_clades.txt')
    parser.add_argument("-aa", "--aa_output", help="Path to save site level conservation statistics for the full tree.", default='site_dnds.csv')
    parser.add_argument("-ex", "--expanded_output", help="Path to save per-mutation translation output.",default="expanded_translation.csv")
    args = parser.parse_args()
    return args

def get_types():
    types = []
    for a in 'ACGT':
        for b in 'ACGT':
            if a != b:
                types.append(a+">"+b)
    return types

def get_mtypes(otdf, types):
    mtypes = {a:0 for a in types}
    for i,d in otdf.iterrows():
        ntv = d.nt_mutations.split(";")
        aas = d.aa_mutations.split(";")
        assert len(ntv) == len(aas)
        for i,m in enumerate(ntv):
            mty = m[0] + ">" + m[-1]
            aa = aas[i].split(":")
            if aa[1][0] == aa[1][-1] and d.leaves_sharing_mutations > 1: #only use synonymous mutations to build the distribution.
                if mty in mtypes:
                    mtypes[mty] += 1
    tm = sum(mtypes.values())
    norm_mtypes = {k:v/tm for k,v in mtypes.items()}
    return norm_mtypes

def build_reference_loc_codons():
    coordmatch = {}
    with open("exons.gtf") as inf:
        for entry in inf:
            spent = entry.strip().split()
            g = spent[9].strip(';"')
            coordmatch[str(int(spent[3])-1) + "-" + spent[4]] = g
    reference_loc_codons = {}
    with open('genes.fasta') as inf:
        current = None
        tstr = ""
        for entry in inf:
            if entry[0] == ">":
                if len(tstr) > 0:
                    ld = {}
                    locn = 1
                    for i in np.arange(0,len(tstr),3):
                        ld[locn] = tstr[i:i+3]
                        locn += 1
                    reference_loc_codons[current] = ld
                new = coordmatch[entry.split("::")[1].split(":")[1].strip()]
                if new != current:
                    tstr = ""
                current = new
            else:
                tstr += entry.strip()
        ld = {}
        locn = 0
        for i in np.arange(0,len(tstr),3):
            ld[locn] = tstr[i:i+3]
            locn += 1
        reference_loc_codons[current] = ld
    return reference_loc_codons
reference_loc_codons = build_reference_loc_codons()

def process_tdf(otdf,cldf):
    #need to reshape otdf to extract the sets of mutations, node occurrences, and associated codon changes
    #going to produce a very lorge dataframe
    tdf = {k:[] for k in ['node_id','AA','NT','CC',"Leaves","Clade"]}
    for i,d in tqdm(otdf.iterrows()):
        nts = d.nt_mutations.split(";")
        ccs = d.codon_changes.split(";")
        for i, aa in enumerate(d.aa_mutations.split(";")):
            tdf['node_id'].append(d.node_id)
            tdf['AA'].append(aa)
            tdf['NT'].append(nts[i])
            tdf['CC'].append(ccs[i])
            tdf['Leaves'].append(d.leaves_sharing_mutations)
            try:
                tdf['Clade'].append(cldf.loc[d.node_id].annotation_1)
            except KeyError:
                tdf['Clade'].append('N/A')
    tdf = pd.DataFrame(tdf)
    tdf['Gene'] = tdf.AA.apply(lambda x:x.split(":")[0])
    tdf['Loc'] = tdf.NT.apply(lambda x:int(x.split(",")[0][1:-1]))
    tdf['MType'] = tdf.NT.apply(lambda x:x[0] + ">" + x[-1])
    tdf['Synonymous'] = tdf.AA.apply(lambda x:(x.split(":")[1][0] == x.split(":")[1][-1]))
    def get_aal(aa):
        aal = aa.split(":")[1]
        return int(aal[1:-1])
    tdf['AAL'] = tdf.AA.apply(get_aal)
    def get_aac(aa):
        aal = aa.split(":")[1]
        return aal[0] + ">" + aal[-1]
    tdf['AAC'] = tdf.AA.apply(get_aac)
    tdf['OG'] = tdf.AAC.apply(lambda x:x[0])
    tdf["AL"] = tdf.AAC.apply(lambda x:x[-1])
    tdf['OGC'] = tdf.CC.apply(lambda x:x.split(">")[0])
    tdf['ALTC'] = tdf.CC.apply(lambda x:x.split(">")[1])
    fr = []
    for i,d in tdf.iterrows():
        try:
            ref_codon = reference_loc_codons[d.Gene][d.AAL]
        except:
            print(d.Gene, d.AAL, d.NT, d.AA)
            fr.append(False)
            continue
        if d.OGC == ref_codon:
            fr.append("FR")
        elif d.ALTC == ref_codon:
            fr.append("Back")
        else:
            fr.append("Other")
    tdf['FromRef'] = fr
    tdf['StopGain'] = tdf.AL.apply(lambda x:(translate[x] == 'Stop'))
    return tdf

def build_expectation(norm_mtypes):
    for k,v in list(aagroups.items()):
        aagroups.update({aa:k for aa in v})

    def get_changeset(codon, mts = norm_mtypes):
        codon = list(codon)
        naas = {}
        for i,b in enumerate(codon):
            for alternative in 'ACGT':
                if b != alternative:
                    ncodon_list = [c for c in codon]
                    ncodon_list[i] = alternative
                    ncodon = ''.join(ncodon_list)
                    #naa = translate[ncodon]
                    naas[ncodon] = mts[b + ">" + alternative] #scale by the probability that this specific change happens as a mutation
        tbp = sum(naas.values()) #rescale so that the distribution is conditioned on a mutation happening at all
        return {k:v/tbp for k,v in naas.items()}
    aaprob = {}
    for codon, aa in translate.items():
        aaprob[codon] = get_changeset(codon)
    codons = list(aaprob.keys())
    valid_aa = set(list(translate.values()))
    aa_tweight = {aa:{aa:0 for aa in valid_aa} for aa in valid_aa}
    for c1 in codons:
        for c2 in codons:
            if c1 != c2:
                a1 = translate[c1]
                a2 = translate[c2]
                aa_tweight[a1][a2] += aaprob[c1].get(c2,0)
    cod_dnds = {}
    for c1, cd in aaprob.items():
        non = 0
        syn = 0
        for c2, p in cd.items():
            aa1 = translate[c1]
            aa2 = translate[c2]
            if aa1 == aa2:
                syn += p
            else:
                non += p
        if syn == 0:
            cod_dnds[c1] = np.inf
        else:
            cod_dnds[c1] = non/syn
    return translate, aaprob, cod_dnds

def process_aadf(tdf,cod_dnds,aaprob,translate):
    #now, for every site, look at the mutations (which are all from the reference codon)
    #get the distribution of proportional changes expected and actual
    #and compute a squared error for the difference across categories
    #and create a dataframe with it. 
    aadf = {k:[] for k in ['Gene','Loc','RefAA','Multinom','Chi2','Count','SynCount','NonCount','DnDs']}#,'Clade']}
    for g, osdf in tdf[tdf.Leaves > 1].groupby("Gene"):
        for l, sdf in osdf.groupby("AAL"):
            refc = reference_loc_codons[g][l] 
            nrat = cod_dnds[refc]
            svc = sdf.Synonymous.value_counts()
            #instead of raw counts, weight by total descendents?
            #on an unbiased tree, this shouldn't affect the ratio, but maybe can be more explanatory
            #svc = np.log10(sdf.groupby("Synonymous").Leaves.sum())
            if True not in svc.index:
                dnds = np.inf
                sync = 0
                nonc = svc[False]
            elif False not in svc.index:
                dnds = 0
                nonc = 0
                sync = svc[True]
            else:
                dnds = (svc[False]/svc[True])/nrat
                nonc = svc[False]
                sync = svc[True]

            mc = sdf.OGC.value_counts().index[0]
            if sdf.OGC.nunique() > 1:
                print(g,l)
                print(sdf.OGC.value_counts())
            assert sdf.OGC.nunique() == 1
            exp_r = aaprob[sdf.OGC.iloc[0]]
            act_r = dict(sdf.ALTC.value_counts())
            aa_v = []
            exp_pv = []
            exp_cv = []
            act_cv = []
            for aa,p in exp_r.items():
                aa_v.append(aa)
                exp_pv.append(p)
                act_cv.append(act_r.get(aa,0))
            total = sum(act_cv)
            for p in exp_pv:
                exp_cv.append(p*total)
            mod = multinomial(total,exp_pv)
            mnom = mod.logpmf(act_cv)
            try:
                chi2 = chisquare(act_cv, exp_cv)[1]
            except:
                chi2 = np.nan
                print("Chi2 got wrecked", act_cv, exp_cv, total)
                print(exp_pv)

            aadf['Gene'].append(g)
            aadf['Loc'].append(l)
            aadf['RefAA'].append(translate[sdf.OGC.iloc[0]])
            aadf['Multinom'].append(mnom)
            aadf['Chi2'].append(chi2)
            aadf['Count'].append(sdf.shape[0])
            aadf['SynCount'].append(sync)
            aadf['NonCount'].append(nonc)
            aadf['DnDs'].append(dnds)
    aadf = pd.DataFrame(aadf)
    aadf['CorrChi2'] = aadf.Chi2*aadf.shape[0]
    aadf['LogDnDs'] = np.log10(aadf['DnDs'])
    return aadf

def write_chimera(locvals, name = 'conservation', out = 'conservation.txt'):
    #write a chimera annotation file at the residue level
    with open(out,"w+") as outf:
        print("attribute: " + name,file=outf)
        print("recipient: residues",file=outf)
        for l,v in locvals:
            print("\t:"+str(l)+"\t"+str(v),file=outf)

def basic_pipe():
    args = parse_basic_args()
    otdf = pd.read_csv(args.translation,sep='\t')
    cldf = pd.read_csv(args.clades,sep='\t').set_index('sample')
    types = get_types()
    norm_mtypes = get_mtypes(otdf, types)
    translate, aaprob, cod_dnds = build_expectation(norm_mtypes)
    tdf = process_tdf(otdf,cldf)
    tdf.to_csv(args.expanded_output,index=False)
    aadf = process_aadf(tdf,cod_dnds,aaprob,translate)
    aadf.to_csv(args.aa_output,index=False)

if __name__ == "__main__":
    basic_pipe()  