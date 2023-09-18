"""
Processing script for the raw 'final' eRASS1 source catalogues.

eRASS Data Validation Team, 2022
"""
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.table import Table
import healsparse as hsp
import json
import numpy as np
import yaml

line_separator = '*********'


def add_erass_name(_df):
    """
    Add in name for each source in the catalogue: e.g. eRASSt_J093641.9+111347

    :param _df: input pandas dataframe
    :return _df: input dataframe with source name column added.
    """
    name_arr = []
    ra_hms = []
    dec_dms = []
    for i, row in _df.iterrows():
        c = SkyCoord(row['RA_1'], row['DEC_1'], frame='icrs', unit='deg')
        name_arr.append('eRASSt_J{0}{1}'.format(c.ra.to_string(unit=u.hourangle, sep='', precision=0, pad=True),
                                                c.dec.to_string(sep='', precision=0, alwayssign=True, pad=True)))
        ra_hms.append(str(c.ra.to_string(u.hour)))
        dec_dms.append(str(c.dec.to_string(u.degree, alwayssign=True)))
    _df['NAME'] = name_arr
    return _df


def dropout_unused_columns(_cat_type, _cat_raw, _cat_proc, _cfg, _log_file, _ver):
    """
    Remove columns containing redundant/ unused information and write processed catalogue to file.

    :param _cat_type: Type of catalogue to process (e.g. threeband vs oneband).
    :param _cat_raw: File name for unprocessed catalogue.
    :param _cat_proc: File name for processed catalogue.
    :param _cfg: Config file for the processing.
    :param _log_file: Output log file for the processing.
    :param _ver: Processing version to use.
    """
    f_unused_cols = '../%s/%s' % (_cfg[_ver]['unused_columns']['dir'], _cfg[_ver]['unused_columns'][_cat_type])
    unused_cols = np.genfromtxt(f_unused_cols, dtype=str)

    # Load in the catalogue, drop out unused columns, save to file
    t = Table.read(_cat_raw, format='fits')
    for unused_col in unused_cols:
        t.remove_column(unused_col)
    t.write(_cat_proc, format='fits', overwrite=True)

    # Log the changes made to file
    with open(_log_file, 'a') as f:
        f.write('%s \n Removing unused columns... \n %s \n' % (line_separator, line_separator))
        for col in unused_cols:
            f.write('Removed %s column \n' % col)


def flag_spurious_sources(_cat, _cat_proc, _cfg, _log_file, _ver):
    """
    Flag spurious sources in the eRASS1 catalogue. The following abbreviations are used in the flagging:
        SNR: supernova remnant
        BPS: bright point source
        SCL: star cluster
        LGA: local extended galaxy
        GC: galaxy cluster
        GC_CONS: galaxy cluster (conservative radius)

    :param _cat: Type of catalogue to process (e.g. threeband vs oneband).
    :param _cat_proc: File name for processed catalogue.
    :param _cfg: Config file for the processing.
    :param _log_file: Output log file for the processing.
    :param _ver: Processing version to use.
    """
    _t = Table.read(_cat_proc, format='fits')

    # Load in HealSparse maps
    nside_coverage = 4096
    hsp_map_galactic = hsp.HealSparseMap.read('../%s' % _cfg[_ver]['healsparse_maps']['galactic'],
                                              nside_coverage=nside_coverage)
    hsp_map_clusters = hsp.HealSparseMap.read('../%s' % _cfg[_ver]['healsparse_maps']['clusters'],
                                              nside_coverage=nside_coverage)
    hsp_map_clusters_conserv = hsp.HealSparseMap.read('../%s' % _cfg[_ver]['healsparse_maps']['clusters_conservative'],
                                                      nside_coverage=nside_coverage)

    # Set up arrays for storing source flags
    len_t = len(_t)
    positions = np.asarray([_t['RA_CORR'], _t['DEC_CORR']]).T
    flag_snr = np.zeros(len_t, dtype=int)
    flag_bps = np.zeros(len_t, dtype=int)
    flag_starcluster = np.zeros(len_t, dtype=int)
    flag_localgal = np.zeros(len_t, dtype=int)
    flag_galcluster = np.zeros(len_t, dtype=int)
    flag_galcluster_cons = np.zeros(len_t, dtype=int)

    # Get source flags now using the healsparse maps
    for i, row in enumerate(positions):
        # Bright point source, SNR flagging
        vals_gal = hsp_map_galactic.get_values_pos(row[0], row[1], lonlat=True)
        flag_snr[i] = vals_gal[1]
        flag_bps[i] = vals_gal[2]
        flag_localgal[i] = vals_gal[4]
        flag_starcluster[i] = vals_gal[5]

        # Galaxy cluster flagging
        flag_galcluster[i] = hsp_map_clusters.get_values_pos(row[0], row[1], lonlat=True)
        flag_galcluster_cons[i] = hsp_map_clusters_conserv.get_values_pos(row[0], row[1], lonlat=True)

    # Tidy up the flags to 1/0
    flag_snr[flag_snr < 1] = 0
    flag_bps[flag_bps < 1] = 0
    flag_localgal[flag_localgal < 1] = 0
    flag_starcluster[flag_starcluster < 1] = 0

    # Set flags
    _t['FLAG_SP_SNR'] = flag_snr
    _t['FLAG_SP_BPS'] = flag_bps
    _t['FLAG_SP_SCL'] = flag_starcluster
    _t['FLAG_SP_LGA'] = flag_localgal
    _t['FLAG_SP_GC'] = flag_galcluster
    _t['FLAG_SP_GC_CONS'] = flag_galcluster_cons

    # Flag also the pathological cases
    _t['FLAG_NO_RADEC_ERR'] = np.where(np.isnan(_t['RADEC_ERR']), 1, 0)
    _t['FLAG_NO_EXT_ERR'] = np.where(np.isnan(_t['EXT_ERR']), 1, 0)
    if (_cat == 'one_band_erass3') or (_cat == 'one_band_erass4') or ((_cat == 'one_band') and (int(_ver) > 7)):
        _t['FLAG_NO_CTS_ERR'] = np.where(np.isnan(_t['ML_CTS_ERR_1']), 1, 0)
    else:
        _t['FLAG_NO_CTS_ERR'] = np.where(np.isnan(_t['ML_CTS_ERR_0']), 1, 0)

    # Write to file
    _t.write(_cat_proc, format='fits', overwrite=True)

    with open(_log_file, 'w') as f:
        f.write('%s \n Flagging spurious sources... \n %s \n' % (line_separator, line_separator))


def correct_positional_errors(_cat_proc, _cfg, _log_file, _ver):
    """
    Correct the RADEC_ERR.

    :param _cat_proc: File name for processed catalogue.
    :param _cfg: Config file for the processing.
    :param _log_file: Output log file for the processing.
    :param _ver: Processing version to use.
    """
    t = Table.read(_cat_proc, format='fits')
    radecerr_scale_factor = _cfg[_ver]['RADEC_ERR_SCALE_FACTOR']
    print('using %s scale factor to correct radec_err' % radecerr_scale_factor)
    t['RADEC_ERR_CORR'] = radecerr_scale_factor * t['RADEC_ERR']

    # Write to file
    t.write(_cat_proc, format='fits', overwrite=True)

    # Log the changes made to file
    with open(_log_file, 'a') as f:
        f.write('%s \n Added in corrected RADEC_ERR_CORR column- %s*RADEC_ERR... \n %s \n' % (line_separator,
                                                                                              radecerr_scale_factor,
                                                                                              line_separator))


def update_column_descriptions_and_units(_cat_proc, _cfg, _log_file, _ver):
    """
    Add in the column descriptions to the source catalogues. Ensure all the column units are correct.

    :param _cat_proc: File name for processed catalogue.
    :param _cfg: Config file for the processing.
    :param _log_file: Output log file for the processing.
    :param _ver: Processing version to use.
    """
    # read in processed catalogue catalogue
    t = Table.read(_cat_proc, format='fits')

    # read in column meta information
    with open("../data/meta_info/core.yml", 'r') as stream:
        meta_core = yaml.safe_load(stream)
    with open("../data/meta_info/noncore.yml", 'r') as stream:
        meta_noncore = yaml.safe_load(stream)

    # updated column information store
    info_store = {}

    # update the core meta info for each column
    for col, meta_info in meta_core.items():
        try:
            info_store[col]['unit'] = meta_info['unit']
            if meta_info['unit'] is not None:
                t[col].unit = meta_info['unit']
        except Exception as e:
            print(e)
        try:
            if len(meta_info['description']) < 68:
                t[col].description = meta_info['description']
                info_store[col] = {}
                info_store[col]['description'] = meta_info['description']
        except Exception as e:
            print(e)

    # update the non-core meta info for each column
    energy_dict = _cfg[_ver]['catalogues'][_cat]['energy_bands']
    for band_id, energy_band in energy_dict.items():
        for col, meta_info in meta_noncore.items():
            col_update = '%s%s' % (col, band_id)
            des_string = '%s %s.' % (meta_info['description'], energy_band)
            try:
                t[col_update].unit = meta_info['unit']
                if len(des_string) < 68:
                    t[col_update].description = des_string
                info_store[col_update] = {}
                info_store[col_update]['unit'] = meta_info['unit']
                info_store[col_update]['description'] = des_string
            except Exception as e:
                print(col_update, e)
    t.write(_cat_proc, format='fits', overwrite=True)

    with open('../data/meta_info/%s.yml' % _cat, 'w') as outfile:
        yaml.dump(info_store, outfile, default_flow_style=False)


def process_catalogue(_cat, _cfg, _ver):
    """
    Run all catalogue processing checks.

    :param _cat: Type of catalogue to process (e.g. threeband vs oneband).
    :param _cfg: Config file for the processing.
    :param _ver: Processing version to use.
    """
    # Define log file for generating the catalogue
    log_file_base = config_data[_ver]['catalogues'][_cat]['proc_log']
    log_file = '../%s%s.txt' % (log_file_base, _ver)
    print(log_file)

    # Output config file used for processing to the log file
    with open(log_file, 'w') as f:
        f.write('%s \n Log of generating a processed eRASS1 catalogue \n %s \n' % (line_separator, line_separator))
        f.write('Program was started with following config file: \n')
        f.write('%s \n' % line_separator)
        f.write(json.dumps(config_data))
        f.write('\n')
        f.write('%s \n' % line_separator)

    # Define the raw and processed catalogues
    cat_raw = '../%s' % _cfg[_ver]['catalogues'][_cat]['raw']
    cat_proc = '../%s%s.fits' % (_cfg[_ver]['catalogues'][_cat]['processed'], _ver)

    # Remove unused columns
    dropout_unused_columns(_cat, cat_raw, cat_proc, _cfg, log_file, _ver)

    # Spurious point source flagging
    flag_spurious_sources(_cat, cat_proc, _cfg, log_file, _ver)

    # Astrometric corrections
    correct_positional_errors(cat_proc, _cfg, log_file, _ver)


if __name__ == '__main__':
    ver = '008'
    print('*** Processed catalogue version: %s***' % ver)

    config_path = '../data/config/config.json'
    with open(config_path) as cjson:
        config_data = json.load(cjson)

    catalogues_to_revise = ['three_band']
    for cat in catalogues_to_revise:
        process_catalogue(cat, config_data, ver)
