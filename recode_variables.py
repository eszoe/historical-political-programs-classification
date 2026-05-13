import pandas as pd
from pathlib import Path

DATA_DIR = Path('data')

corpus = pd.read_csv(DATA_DIR / 'corpus_ger_cze.csv')


def recode_corpus(corpus: pd.DataFrame) -> pd.DataFrame:
    """
    Renames raw columns to short codes, creates directional (pos/neg)
    and combined (topic-level) columns, and returns a clean subset.
    """

    #  Basic columns 
    corpus['start']    = corpus['Anfang']
    corpus['end']      = corpus['Ende']
    corpus['sentence'] = corpus['Segment']

    corpus = corpus.drop(columns=['Code', 'Gewicht', 'Dokumentgruppe'], errors='ignore')

    # Directional categories (pos/neg)
    corpus['milit_pos']       = corpus['militarism pos']
    corpus['milit_neg']       = corpus['militarism neg']
    corpus['conscrip_pos']    = corpus['conscription pos']
    corpus['conscrip_neg']    = corpus['conscription neg']
    corpus['cit_army_pos']    = corpus['citizen army pos']
    corpus['democ_pos']       = corpus['democracy pos']
    corpus['democ_neg']       = corpus['democracy neg']
    corpus['elec_inc_pos']    = corpus['electoral inclusion pos']
    corpus['elec_inc_neg']    = corpus['electoral inclusion neg']
    corpus['parl_aut_pos']    = corpus['parliamentary autonomy pos']
    corpus['civ_lib_pos']     = corpus['civil liberties pos']
    corpus['civ_lib_neg']     = corpus['civil liberties neg']
    corpus['rule_law_pos']    = corpus['rule of law pos']
    corpus['rule_law_neg']    = corpus['rule of law neg']
    corpus['spec_hun_pos']    = corpus['special status Hungary pos']
    corpus['spec_hun_neg']    = corpus['special status Hungary neg']
    corpus['dec_fed_pos']     = corpus['federalism/decentralization pos']
    corpus['dec_fed_neg']     = corpus['federalism/decentralization neg']
    corpus['sec_pos']         = corpus['secession pos']
    corpus['sec_neg']         = corpus['secession neg']
    corpus['capit_pos']       = corpus['capitalism pos']
    corpus['capit_neg']       = corpus['capitalism neg']
    corpus['free_trade_pos']  = corpus['free trade pos']
    corpus['free_trade_neg']  = corpus['free trade neg']
    corpus['exploit_neg']     = corpus['exploitation, neg']
    corpus['growth_pos']      = corpus['growth pos'] + corpus['growth/industrialisation pos']
    corpus['growth_neg']      = corpus['growth neg']
    corpus['pub_debt_neg']    = corpus['public debt neg']
    corpus['int_rate_pos']    = corpus['lower interest rates pos']
    corpus['int_rate_neg']    = corpus['lower interest rates neg']
    corpus['tax_pos']         = corpus['taxes pos']
    corpus['tax_neg']         = corpus['taxes neg']
    corpus['corpor_pos']      = corpus['corporatism pos']
    corpus['coop_pos']        = corpus['cooperatives pos']
    corpus['coop_neg']        = corpus['cooperatives neg']
    corpus['trade_uni_pos']   = corpus['trade unions pos']
    corpus['pub_infra_pos']   = corpus['public infrastructure pos']
    corpus['soc_eq_pos']      = corpus['social equality pos']
    corpus['soc_eq_neg']      = corpus['social equality neg']
    corpus['soc_sec_pos']     = corpus['social security pos']
    corpus['soc_sec_neg']     = corpus['social security neg']
    corpus['pub_heal_pos']    = corpus['public health pos']
    corpus['pub_educ_pos']    = corpus['public education pos']
    corpus['pub_sub_pos']     = corpus['public subsidies pos']
    corpus['pub_sub_neg']     = corpus['public subsidies neg']
    corpus['class_conf_pos']  = corpus['class conflict pos']
    corpus['class_conf_neg']  = corpus['class conflict neg']
    corpus['tra_rel_val_pos'] = corpus['traditional/relig. values pos']
    corpus['tra_rel_val_neg'] = corpus['traditional/relig. values neg']
    corpus['sep_ch_st_pos']   = corpus['separation church state, pos']
    corpus['sep_ch_st_neg']   = corpus['separation church state, neg']
    corpus['nation_pos']      = corpus['nationalism pos']
    corpus['nation_neg']      = corpus['nationalism neg']
    corpus['lang_right_pos']  = corpus['language rights pos'] + corpus['special language rights pos']
    corpus['lang_right_neg']  = corpus['language rights neg'] + corpus['special language rights neg']
    corpus['group_rep_pos']   = corpus['group representation pos']
    corpus['group_rep_neg']   = corpus['group representation neg']
    corpus['immig_pos']       = corpus['immigration pos']
    corpus['immig_neg']       = corpus['immigration neg']
    corpus['agri_pos']        = corpus['agriculture pos']
    corpus['unemp_neg']       = corpus['unemployment neg']

    # Combined topic columns (pos + neg merged)
    corpus['milit']       = corpus['milit_pos']      + corpus['milit_neg']
    corpus['conscrip']    = corpus['conscrip_pos']   + corpus['conscrip_neg']
    corpus['cit_army']    = corpus['cit_army_pos']
    corpus['democ']       = corpus['democ_pos']      + corpus['democ_neg']
    corpus['elec_inc']    = corpus['elec_inc_pos']   + corpus['elec_inc_neg']
    corpus['parl_aut']    = corpus['parl_aut_pos']
    corpus['civ_lib']     = corpus['civ_lib_pos']    + corpus['civ_lib_neg']
    corpus['rule_law']    = corpus['rule_law_pos']   + corpus['rule_law_neg']
    corpus['spec_hun']    = corpus['spec_hun_pos']   + corpus['spec_hun_neg']
    corpus['dec_fed']     = corpus['dec_fed_pos']    + corpus['dec_fed_neg']
    corpus['sec']         = corpus['sec_pos']        + corpus['sec_neg']
    corpus['capit']       = corpus['capit_pos']      + corpus['capit_neg']
    corpus['free_trade']  = corpus['free_trade_pos'] + corpus['free_trade_neg']
    corpus['exploit']     = corpus['exploit_neg']
    corpus['growth']      = corpus['growth_pos']     + corpus['growth_neg']
    corpus['pub_debt']    = corpus['pub_debt_neg']
    corpus['int_rate']    = corpus['int_rate_pos']   + corpus['int_rate_neg']
    corpus['tax']         = corpus['tax_pos']        + corpus['tax_neg']
    corpus['corpor']      = corpus['corpor_pos']
    corpus['coop']        = corpus['coop_pos']       + corpus['coop_neg']
    corpus['trade_uni']   = corpus['trade_uni_pos']
    corpus['pub_infra']   = corpus['pub_infra_pos']
    corpus['soc_eq']      = corpus['soc_eq_pos']     + corpus['soc_eq_neg']
    corpus['soc_sec']     = corpus['soc_sec_pos']    + corpus['soc_sec_neg']
    corpus['pub_heal']    = corpus['pub_heal_pos']
    corpus['pub_educ']    = corpus['pub_educ_pos']
    corpus['pub_sub']     = corpus['pub_sub_pos']    + corpus['pub_sub_neg']
    corpus['class_conf']  = corpus['class_conf_pos'] + corpus['class_conf_neg']
    corpus['tra_rel_val'] = corpus['tra_rel_val_pos']+ corpus['tra_rel_val_neg']
    corpus['sep_ch_st']   = corpus['sep_ch_st_pos']  + corpus['sep_ch_st_neg']
    corpus['nation']      = corpus['nation_pos']     + corpus['nation_neg']
    corpus['lang_right']  = (
        corpus['lang_right_pos'] + corpus['lang_right_neg']
    )
    corpus['group_rep']   = corpus['group_rep_pos']  + corpus['group_rep_neg']
    corpus['immig']       = corpus['immig_pos']      + corpus['immig_neg']
    corpus['agri']        = corpus['agri_pos']
    corpus['unemp']       = corpus['unemp_neg']
    corpus['oth_issue']   = corpus['Other issue']

    #  Select final columns
    cols = [
        'doc_title', 'start', 'end', 'sentence',
        'milit_pos', 'milit_neg',
        'conscrip_pos', 'conscrip_neg',
        'cit_army_pos',
        'democ_pos', 'democ_neg',
        'elec_inc_pos', 'elec_inc_neg',
        'parl_aut_pos',
        'civ_lib_pos', 'civ_lib_neg',
        'rule_law_pos', 'rule_law_neg',
        'spec_hun_pos', 'spec_hun_neg',
        'dec_fed_pos', 'dec_fed_neg',
        'sec_pos', 'sec_neg',
        'capit_pos', 'capit_neg',
        'free_trade_pos', 'free_trade_neg',
        'exploit_neg',
        'growth_pos', 'growth_neg',
        'pub_debt_neg',
        'int_rate_pos', 'int_rate_neg',
        'tax_pos', 'tax_neg',
        'corpor_pos',
        'coop_pos', 'coop_neg',
        'trade_uni_pos',
        'pub_infra_pos',
        'soc_eq_pos', 'soc_eq_neg',
        'soc_sec_pos', 'soc_sec_neg',
        'pub_heal_pos',
        'pub_educ_pos',
        'pub_sub_pos', 'pub_sub_neg',
        'class_conf_pos', 'class_conf_neg',
        'tra_rel_val_pos', 'tra_rel_val_neg',
        'sep_ch_st_pos', 'sep_ch_st_neg',
        'nation_pos', 'nation_neg',
        'lang_right_pos', 'lang_right_neg',
        'group_rep_pos', 'group_rep_neg',
        'immig_pos', 'immig_neg',
        'agri_pos',
        'unemp_neg',
        'milit', 'conscrip', 'cit_army', 'democ', 'elec_inc',
        'parl_aut', 'civ_lib', 'rule_law', 'spec_hun', 'dec_fed',
        'sec', 'capit', 'free_trade', 'exploit', 'growth',
        'pub_debt', 'int_rate', 'tax', 'corpor', 'coop',
        'trade_uni', 'pub_infra', 'soc_eq', 'soc_sec',
        'pub_heal', 'pub_educ', 'pub_sub', 'class_conf',
        'tra_rel_val', 'sep_ch_st', 'nation', 'lang_right',
        'group_rep', 'immig', 'agri', 'unemp', 'oth_issue'
    ]

    # Safety check — warn if any expected column is missing
    missing = [c for c in cols if c not in corpus.columns]
    if missing:
        print(f"⚠️  Missing columns: {missing}")

    return corpus[cols].copy()


corpus_rec = recode_corpus(corpus)
print(f"Shape: {corpus_rec.shape}")
print(corpus_rec.columns.tolist())

# Save with a new name
corpus_rec.to_csv(DATA_DIR / 'corpus_rec.csv', index=False)