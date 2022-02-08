import pandas as pd
import numpy as np
import argparse
import bindingcalculator as bc
from process_basic import *
from scipy.stats import binom
from scipy.stats import fisher_exact as fe

translate = {'TTT':'F','TTC':'F','TTA':'L','TTG':'L','TCT':'S','TCC':'S','TCA':'S','TCG':'S',
            'TAT':'Y','TAC':'Y','TAA':'Stop','TAG':'Stop','TGT':'C','TGC':'C','TGA':'Stop','TGG':'W',
            'CTT':'L','CTC':'L','CTA':'L','CTG':'L','CCT':'P','CCC':'P','CCA':'P','CCG':'P',
            'CAT':'H','CAC':'H','CAA':'Q','CAG':'Q','CGT':'R','CGC':'R','CGA':'R','CGG':'R',
        'ATT':'I','ATC':'I','ATA':'I','ATG':'M','ACT':'T','ACC':'T','ACA':'T','ACG':'T',
            'AAT':'N','AAC':'N','AAA':'K','AAG':'K','AGT':'S','AGC':'S','AGA':'R','AGG':'R',
            'GTT':'V','GTC':'V','GTA':'V','GTG':'V','GCT':'A','GCC':'A','GCA':'A','GCG':'A',
            'GAT':'D','GAC':'D','GAA':'E','GAG':'E','GGT':'G','GGC':'G','GGA':'G','GGG':'G'}
aagroups = {'aliphatic':['G','A','V','L','I'], 'hydroxyl':['S','C','U','T','M'], 'cyclic':['P'], 'aromatic':['F','Y','W'], 'basic':['H','K','R'], 'acidic':['D','E','N','Q'], 'N/A':['Stop']}   

def parse_epistasis_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--translation", help="Path to a translation table produced by matUtils extract.", default = "")
    parser.add_argument("-et", "--expanded_translation",help='Path to a processed, per-mutation translation table. Code will generate one if not provided.',default = None)
    parser.add_argument("-g", "--gtf", help="Path to a gtf containing exon spans", default='exons.gtf')
    parser.add_argument("-f", "--fasta", help="Path to a fasta containing the gene sequences matching the exons", default='genes.fasta')
    parser.add_argument("-c", "--clades", help="Path to a file containing node and clade matching annotations.", required=True)
    parser.add_argument("-ex", "--expanded_output", help="Path to save per-mutation translation output.",default="expanded_translation.csv")
    parser.add_argument("-pa", "--pair_epistasis", help="Path to save pairwise epistasis output.",default="pair_epistasis.csv")
    parser.add_argument("-aa", "--aa_output", help="Path to save site level conservation statistics for pair anchored sites.", default='binding_dnds.csv')
    parser.add_argument("-p", '--paths', help='Path to all sample path text file to accumulate mutations from.',required=True)
    parser.add_argument("-b", '--binding', help='Path to binding calculator data file.',default='escape_calculator_data.csv')
    parser.add_argument("-th", '--threshold', type=int,help='Set a minimum count of leaves downstream to consider a mutation as anchor. Default 1000',default=1000)
    args = parser.parse_args()
    return args

def get_npd(pathfile):
    #build the tree as a dictionary with recursive entries so parents can be accessed in O(1) time
    npd = {}
    skipped = 0
    with open(pathfile) as inf:
        for entry in inf:
            spent = entry.strip().split("\t")
            if len(spent) == 1:
                skipped += 1
                continue #nodes from the root, with no parent mutations, will have no entries
            last = None
            for n in spent[1].split():
                nname, mutations = n.split(":")
                if nname not in npd and last != None:
                    npd[nname] = last
                last = nname
            if spent[0] != last:
                npd[spent[0]] = last
    return npd

#for each mutation, for each upstream site of note, compute whether that mutation exhibits a difference in syn/non frequencies 
#downstream vs not downstream of it. 
#this is essentially a very limited pairwise epistasis scan in a brute force fashion
#if we can identify interesting patterns here, putting effort into developing a more subtle/efficient algorithm
#may be worth an investment. Or, you know, queue up some brute force and let it run for a long long time. lol.
def get_dnds(g,l,sdf,cod_dnds):
    if sdf.shape[0] == 0:
        return 0,0,np.nan
    refc = reference_loc_codons[g][l] 
    nrat = cod_dnds[refc]
    svc = sdf.Synonymous.value_counts()
        
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
    return dnds, sync, nonc

def get_binomial_leaves(sdf):
    if sdf.shape[0] == 0:
        return 0, 0, np.nan
    stc = 0
    syn_tvc = sdf[sdf.Synonymous].Leaves.value_counts(normalize=True)
    if 1 in syn_tvc.index:
        stc = syn_tvc[1]
    ntc = 0
    non_tvc = sdf[~sdf.Synonymous].Leaves.value_counts()
    if 1 in non_tvc.index:
        ntc = non_tvc[1]
    #record the binomial probability that this nonsynonymous mutation didn't come from a mean of 0.6
    #could also test for synonymous, but don't really need to if this test is appropriately sensitive
    tl = sdf[~sdf.Synonymous].shape[0]
    if tl == 0:
        return stc, ntc, np.nan
    binom_p = binom.pmf(n=tl, p=0.6, k=ntc)
    ntc = ntc/tl
    return stc, ntc, binom_p

def get_chisquare(subdf,aaprob):
    if subdf.shape[0] == 0:
        return np.nan
    #chi2 of whether the alternatives are unusual
    mc = subdf.OGC.value_counts().index[0]
    #if sdf.OGC.nunique() > 1:
        #print(sdf.OGC)
    #    return np.nan
    assert subdf.OGC.nunique() == 1
    exp_r = aaprob[subdf.OGC.iloc[0]]
    act_r = dict(subdf.ALC.value_counts())
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
    try:
        chi2 = chisquare(act_cv, exp_cv)[1]
    except:
        chi2 = np.nan
    return chi2

def build_bindd(tdf,npd):
    #reduce bind tdf down further into something faster to search. 
    #instead of having a node associated only with its own actual site mutations, traverse back from each of these nodes via npd
    #that way when I go to assign states later, I only have to find the closest parent and can halt early
    #exchanging memory for runtime.
    bind_d = {}
    #temporary subset of tdf just for identifying hierarchies of antibody binding changes
    bind_tdf = tdf[(~tdf.IsLeaf) & (~tdf.Synonymous)]
    aals = bind_tdf.AA
    for i in aals.index:
        #for every node which has some binding changes, start with those changes

        aas_accumulated = list(aals[[i]])
        sites_accumulated = [int(i[3:-1]) for i in aas_accumulated]
        #bind_d[i] = list(aals[[i]])
        cn = i
        #then traverse backwards from it, identifying any ancestors which also have binding changes and storing those as well
        #to speed future calculations
        while True:
            parent = npd.get(cn,None)
            if parent == None:
                break
            elif parent in aals.index:
                #check each of the parent changes
                #ignore any redundant with changes already seen (later on the tree)
                for a in aals[[parent]]:
                    site = int(a[3:-1])
                    if site not in sites_accumulated:
                        sites_accumulated.append(site)
                        aas_accumulated.append(a)
            cn = parent
        #before processing to store results, throw out any aas back to the reference state
        #this allows reversion, or for individual lineages to randomly lose escape power
        #this would allow me to see whether delta is fighting to stay in place, implying that sera escape is still in play
        #or whether it's gone full neutral from the point of origin
        bind_d[i] = [int(l[3:-1]) for l in aas_accumulated if translate[reference_loc_codons['S'][int(l[3:-1])]] != l[-1]]
    return bind_d    

def accumulate_changes(npd,bind_d):
    downstream_of = {}
    for n in npd.keys():
        cn = n
        if n in bind_d:
            downstream_of[n] = bind_d[n]
        else:
            nodes_encountered = []
            while True:
                parent = npd.get(cn, None)
                assert cn != parent #just checking that I don't get caught in an infinite loop because of a malformed tree structure.
                if parent == None:
                    #if you get back to the root without seeing any relevant binding change nodes, its 1.
                    #node_escape_scores[n] = 1 
                    downstream_of[n] = []
                    break
                elif parent in bind_d:
                    #if you reach a node with an accumulated set of binding mutations, that determines the score of this sample
                    #and everything along the way to it
                    #score = calculator.binding_retained(bind_d[parent])
                    state = bind_d[parent]
                    downstream_of[parent] = state
                    for an in nodes_encountered:
                        downstream_of[an] = state
                    break
                elif parent in downstream_of:
                    state = downstream_of[parent]
                    downstream_of[n] = state
                    for an in nodes_encountered:
                        downstream_of[an] = state
                    break
                nodes_encountered.append(cn)
                cn = parent
    return downstream_of

def assign_epicols(tdf,downstream_of,thresh = 1000):
    tdf['AccumulatedSites'] = tdf.index.map(lambda x:downstream_of.get(x,[]))
    site_counts = {}
    for ast in tdf.AccumulatedSites:
        for l in ast:
            if l not in site_counts:
                site_counts[l] = 0
            site_counts[l] += 1
    #convert the accumulated sites into a one-hot encoding (at least of the most common sites)
    targets = [k for k,v in site_counts.items() if v > thresh]
    encode_cols = {t:[] for t in targets}
    for ast in tdf.AccumulatedSites:
        for t in encode_cols:
            if t in ast:
                encode_cols[t].append(True)
            else:
                encode_cols[t].append(False)
    for k,v in encode_cols.items():
        tdf["Down" + str(k)] = v
    return tdf, encode_cols

def process_pair_aadf(tdf,cod_dnds,aaprob,translate,calculator,encode_cols):
    #now, for every site, look at the mutations (which are all from the reference codon)
    #get the distribution of proportional changes expected and actual
    #and compute a squared error for the difference across categories
    #and create a dataframe with it. 
    aadf = {k:[] for k in ['Gene','Loc','RefAA','Chi2','Count','SynCount','SynSingleLeaves','NonCount','NonSingleLeaves','NonBinom','StopCount','DnDs','PairedWithSite','PairedState']}
    #aadf.update({tc:[] for tc in encode_cols.keys()})
    #for clade, outdf in tdf[(tdf.Leaves > 1) & (tdf.FromRef == 'FR')].groupby("Clade"):
    #    for fourdown, ootdf in outdf.groupby("Post484"):
    #for g, osdf in tdf[(tdf.FromRef == 'FR')].groupby("Gene"):
    g = 'S'
    for l, osdf in tdf[(tdf.FromRef == 'FR') & (tdf.AAL.isin(calculator.sites))].groupby("AAL"):
        actual = reference_loc_codons['S'][l]
        #print(l,osdf.OGC.value_counts())
        osdf = osdf[osdf.OGC == actual]
        if osdf.shape[0] == 0:
            continue
        for matchsite in encode_cols.keys():
            sdf = osdf[osdf['Down'+str(matchsite)]]
            #getting stop counts is straightforward.
            stop_c = sdf[sdf.StopGain].shape[0]
            #use functions to fetch various statistics
            stc, ntc, binom_p = get_binomial_leaves(g,l,sdf)
            dnds, sync, nonc = get_dnds(g,l,sdf,cod_dnds)
            chi2 = get_chisquare(sdf,aaprob)
            
            aadf['Gene'].append(g)
            aadf['Loc'].append(l)
            aadf['RefAA'].append(translate[reference_loc_codons['S'][l]])
            aadf['Chi2'].append(chi2)
            aadf['Count'].append(sdf.shape[0])
            aadf['SynCount'].append(sync)
            aadf['SynSingleLeaves'].append(stc)
            aadf['NonCount'].append(nonc)
            aadf['NonBinom'].append(binom_p)
            aadf['NonSingleLeaves'].append(ntc)
            aadf['StopCount'].append(stop_c)
            aadf['DnDs'].append(dnds)
            aadf['PairedWithSite'].append(matchsite)
            aadf['PairedState'].append(True)

            #repeat the above but NOT downstream of the matched site
            #so the other part of the osdf dataframe.
            sdf = osdf[~osdf['Down'+str(matchsite)]]
            #getting stop counts is straightforward.
            stop_c = sdf[sdf.StopGain].shape[0]
            #use functions to fetch various statistics
            stc, ntc, binom_p = get_binomial_leaves(g,l,sdf)
            dnds, sync, nonc = get_dnds(g,l,sdf,cod_dnds)
            chi2 = get_chisquare(sdf,aaprob)
            aadf['Gene'].append(g)
            aadf['Loc'].append(l)
            aadf['RefAA'].append(translate[reference_loc_codons['S'][l]])
            aadf['Chi2'].append(chi2)
            aadf['Count'].append(sdf.shape[0])
            aadf['SynCount'].append(sync)
            aadf['SynSingleLeaves'].append(stc)
            aadf['NonCount'].append(nonc)
            aadf['NonBinom'].append(binom_p)
            aadf['NonSingleLeaves'].append(ntc)
            aadf['StopCount'].append(stop_c)
            aadf['DnDs'].append(dnds)
            aadf['PairedWithSite'].append(matchsite)
            aadf['PairedState'].append(False)

    aadf = pd.DataFrame(aadf)
    return aadf

def build_pairdf(aadf,calculator):
    pairdf = {k:[] for k in ['target','anchor','inc','outc','fepv','indnds','outdnds','inbind','outbind','anchorbind']}
    for tl, osdf in aadf.groupby("Loc"):
        for al, sdf in osdf.groupby("PairedWithSite"):
            if al == tl:
                continue
            sdf = sdf.set_index("PairedState")
            tps = sdf.loc[True]
            fps = sdf.loc[False]
            if tps.Count > 10 and fps.Count > 10:
                pairdf['target'].append(tl)
                pairdf['anchor'].append(al)
                pairdf['inc'].append(tps.Count)
                pairdf['outc'].append(fps.Count)
                pv = fe([[tps.NonCount,tps.SynCount],[fps.NonCount,fps.SynCount]])[1]
                pairdf['fepv'].append(pv)
                pairdf['indnds'].append(tps.DnDs)
                pairdf['outdnds'].append(fps.DnDs)
                pairdf['inbind'].append(calculator.binding_retained([tl,al]))
                pairdf['outbind'].append(calculator.binding_retained([tl]))
                pairdf['anchorbind'].append(calculator.binding_retained([al]))
    pairdf = pd.DataFrame(pairdf)

def epistasis_pipe():
    args = parse_epistasis_args()
    otdf = pd.read_csv(args.translation,sep='\t')
    cldf = pd.read_csv(args.clades,sep='\t').set_index("sample")
    types = get_types()
    norm_mtypes = get_mtypes(otdf, types)
    translate, aaprob, cod_dnds = build_expectation(norm_mtypes)
    print("Reshaping translation to per-mutation.")
    tdf = process_tdf(otdf,cldf)
    tdf['StopGain'] = tdf.ALC.apply(lambda x:(translate[x] == 'Stop'))
    tdf.to_csv(args.expanded_output,index=False)
    print("Ascertaining binding across translation.")
    npd = get_npd(args.paths)
    calculator = bc.BindingCalculator(csv_or_url=args.binding,eliciting_virus='SARS-CoV-2',source_lab='all',neutralizes_Omicron='either',metric='sum of mutations at site',mutation_escape_strength=1)
    bind_d = build_bindd(tdf,npd)
    downstream_of = accumulate_changes(npd,bind_d)
    tdf, encode_cols = assign_epicols(tdf,downstream_of,args.thresh)
    print("Building amino-acid level conservation analysis with pair anchors.")
    aadf = process_pair_aadf(tdf,cod_dnds,aaprob,translate,calculator,encode_cols)
    aadf.to_csv(args.aa_output,index=False)
    print("Performing statistical testing and reshaping final output.")
    pairdf = build_pairdf(aadf,calculator)
    pairdf['corrfepv'] = pairdf.fepv * pairdf.shape[0]
    pairdf.to_csv(args.pair_epistasis,index=False)
    print("Complete.")

if __name__ == "__main__":
    epistasis_pipe()  