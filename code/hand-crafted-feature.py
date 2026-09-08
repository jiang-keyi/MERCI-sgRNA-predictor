import pandas
import time
import sklearn
import numpy as np
import Bio.SeqUtils as SeqUtil
import Bio.Seq as Seq
import math
import sys
import Bio.SeqUtils.MeltingTemp as Tm
import pickle
import itertools
from multiprocessing import Pool


#structural feature
feature_options = {
                 "testing_non_binary_target_name": 'ranks',
                 'include_pi_nuc_feat': True,
                 "gc_features": True,
                 "nuc_features": True,
                 "include_Tm": True,
                 "include_structure_features": True,
                 "order": 3,
                 "num_proc": 20,
                 "normalize_features":None
                 }
def generate_bytes_file(data):
    scaffold_seq = "GTTTTAGAGCTAGAAATAGCAAGTTAAAATAAGGCTAGTCCGTTATCAACTTGAAAAAGTGGCACCGAGTCGGTGCTTT";
    gRNA_bytes = '\n'.join( list( data.apply( lambda x: '>{0}\n{1}'.format( x[:21], x[:20] ) ) ) ).encode()
    gRNA_plus_tracr_bytes = '\n'.join(
        list( data.apply( lambda x: '>{0}\n{1}'.format( x[:21], x[:20] + scaffold_seq ) ) ) ).encode()
    return gRNA_bytes, gRNA_plus_tracr_bytes

def get_structural_feat(rows):
    bytes_list = generate_bytes_file( rows )
    # print('Get bytes_list 0,20bp')
    # 20bp input for free energy feaure
    lst_0 = get_dG( bytes_list )[0].split( '\n' )
    # 99bp input（20bp + 79bp tracr）
    # print('Get bytes_list 1,99bp')
    lst_1 = get_dG( bytes_list )[1].split( '\n' )
    r = []
    '''
    a--('>AAAAAACACAAGCAAGACCG', '>AAAAAACACAAGCAAGACCG'),fasta header
    b--('AAAAAACACAAGCAAGACCG', 'AAAAAACACAAGCAAGACCGGUUUUAG...AGCUAGAAAUA'), RNAFold transfromed gRNA
    c--('.................... (  0.00)', '........((....(((((((((((((.(((。。。。 (-27.60)')，secondary structure
    '''
    base_pair_List = []

    for a, b, c in grouped( zip( lst_0, lst_1 ), 3 ):
        align_seq = c[1][:99]
        base_pair_List.append( align_seq.replace( '.', 'D' ).replace( '(', 'B' ).replace( ')', 'B' ) )

        # whether exists stem-loop
        ext_stem = "(((((((((.((((....))))...)))))))"
        aligned_stem = align_seq[18:18 + len( ext_stem )]
        stem = 1 if ext_stem == aligned_stem else 0
        dG = c[0].split( ' (' )[1][:-2].strip()
        dG_binding_20 = dG_binding( a[0][1:21] )
        dg_binding_7to20 = dG_binding( a[0][8:21] )
        simple_feature_group = [stem, dG, dG_binding_20, dg_binding_7to20]
        r.append( simple_feature_group )

    ba_rows = pandas.Series( base_pair_List )
    df_feat = pandas.DataFrame( r ).astype( 'float64' )
    colums_name_list = ['stem', 'dG', 'dG_binding_20', 'dg_binding_7to20']
    df_feat.columns = colums_name_list

    return rows, df_feat, ba_rows

def grouped(iterable, n):
    "s -> (s0,s1,s2,...sn-1), (sn,sn+1,sn+2,...s2n-1), (s2n,s2n+1,s2n+2,...s3n-1), ..."
    return zip( *[iter( iterable )] * n )


def base_accessibility(align_seq):
    alignment = align_seq.replace( '.', 'D' ).replace( '(', 'B' ).replace( ')', 'B' )
    r = []
    for i, v in enumerate( alignment ):
        if i in feature_options['secondary_structure_list']:
            r.append( v )
    ext_stem = "(((((((((.((((....))))...)))))))"
    aligned_stem = align_seq[18:18 + len( ext_stem )]
    if ext_stem == aligned_stem:
        r.append( 1 )
    else:
        r.append( 0 )
    return r


def get_dG(bytes_list, RNAfold_BIN='RNAfold'):
    import subprocess
    CMD = [RNAfold_BIN, '--noPS']
    CMD = ' '.join( str( v ) for v in CMD )
    r = []
    for data in bytes_list:
        p = subprocess.Popen( CMD,
                              shell=True,
                              stdin=subprocess.PIPE,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE,
                              executable='/bin/bash' )
        stdout, stderr = p.communicate( input=data )
        stdout = stdout.decode( 'utf-8' )
        stderr = stderr.decode( 'utf-8' )
        r.append( stdout )
    return r


def dG_binding(seq):
    seq = seq.lower()
    dG = {'aa': -0.2, 'tt': -1, 'at': -0.9,
          'ta': -0.6, 'ca': -1.6, 'tg': -0.9,
          'ct': -1.8, 'ag': -0.9, 'ga': -1.5,
          'tc': -1.3, 'gt': -2.1, 'ac': -1.1,
          'cg': -1.7, 'gc': -2.7, 'gg': -2.1, 'cc': -2.9}

    seq = seq.replace( 'u', 't' )
    binding_dG = 0
    dGi = 3.1
    for i in range( 0, len( seq ) - 1 ):
        key = seq[i:i + 2]
        binding_dG += dG[key]
    binding_dG += dGi
    return binding_dG



def countGC(s, length_audit=True):   #GC content features
    '''
    GC content for only the 20mer, as per the Doench paper/code
    '''
    if length_audit:
        assert len( s ) == 21, "seems to assume 21mer"
    return len( s[0:20].replace( 'A', '' ).replace( 'T', '' ) )

def gc_cont(seq):
    return (seq.count( 'G' ) + seq.count( 'C' )) / float( len( seq ) )

def gc_count(data, audit=True):
    gc_count = data.apply( lambda seq: countGC( seq, audit ) )
    gc_count.name = 'GC count'
    return np.array(gc_count)

def gc_above_10(data, audit=True):
    gc_count = data.apply( lambda seq: countGC( seq, audit ) )
    gc_count.name = 'GC count'
    gc_above_10 = (gc_count > 10) * 1
    gc_above_10.name = 'GC > 10'
    return gc_above_10



def Tm_feature(data, feature_options=None):     #thermaodynamics features

    if feature_options is None or 'Tm segments' not in feature_options.keys():
        segments = [(15, 21), (4, 13), (0, 4)]
    else:
        segments = feature_options['Tm segments']

    sequence = data.values
    featarray = np.ones( (sequence.shape[0], 4) )

    for i, seq in enumerate( sequence ):
        rna = False
        featarray[i, 0] = Tm.Tm_staluc( seq, rna=rna )  # 21mer Tm
        featarray[i, 1] = Tm.Tm_staluc( seq[segments[0][0]:segments[0][1]],
                                        rna=rna )  # 5nts immediately proximal of the NGG PAM
        featarray[i, 2] = Tm.Tm_staluc( seq[segments[1][0]:segments[1][1]], rna=rna )  # 8-mer
        featarray[i, 3] = Tm.Tm_staluc( seq[segments[2][0]:segments[2][1]], rna=rna )  # 4-mer

    feat = pandas.DataFrame( featarray, index=data.index,
                             columns=["Tm global_%s" % rna, "5mer_end_%s" % rna, "8mer_middle_%s" % rna,
                                      "4mer_start_%s" % rna] )
    return feat



def zCurve(x):        #Zcurve
    a=[]
    for i in x:
        t = []
        T = i.count('T');
        A = i.count('A');
        C = i.count('C');
        G = i.count('G');
        x_ = (A + G) - (C + T)
        y_ = (A + C) - (G + T)
        z_ = (A + T) - (C + G)
        t.append(x_);
        t.append(y_);
        t.append(z_)
        np.array(a.append(t))
    return a



def atgcRatio(x):       #(A+T)/(C+G)ratio
    a = []
    for i in x:
        t = []
        T = i.count('T')
        A = i.count('A');
        C = i.count('C');
        G = i.count('G');
        t.append((A + T) / (G + C))
        np.array(a.append(t))
    return a


def cumulativeSkew(x):      #GC/AT skew
    a = []
    for i in x:
        t = []
        T = i.count('T')
        A = i.count('A');
        C = i.count('C');
        G = i.count('G');
        GCSkew = (G - C) / (G + C)
        ATSkew = (A - T) / (A + T)
        t.append(GCSkew)
        t.append(ATSkew)
        np.array(a.append(t))
    return a