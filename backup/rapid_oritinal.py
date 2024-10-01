#!/usr/bin/env python3
# *****************************************************************************
# rapid.py
# *****************************************************************************

# Purpose:
# This program is a Python implementation of the core capabilities of the RAPID
# model. RAPID was initially developed using Fortran 90 and the PETSc parallel
# computing library.

# Author:
# Cedric H. David, 2024-2024


# *****************************************************************************
# Import Python modules
# *****************************************************************************
import csv
import netCDF4
import numpy
import os
import pandas as pd
import time

from datetime import datetime, timezone
from scipy.sparse import csc_matrix, diags, identity
from scipy.sparse.linalg import spsolve, spsolve_triangular, factorized, splu
# from scikits.umfpack import splu as umlu


# *****************************************************************************
# Inputs
# *****************************************************************************

# -----------------------------------------------------------------------------
# Test
# -----------------------------------------------------------------------------
con_csv = '../input/Test/rapid_connect_Test.csv'
m3r_ncf = '../input/Test/m3_riv_Test.nc'
kpr_csv = '../input/Test/k_Test.csv'
xpr_csv = '../input/Test/x_Test.csv'

bas_csv = '../input/Test/riv_bas_id_Test.csv'

ZS_dtR = 900

Qou_ncf = '../output/Test/Qout_Test.nc'


# -----------------------------------------------------------------------------
# San_Guad_JHM
# -----------------------------------------------------------------------------
# inp_fld = '../input/San_Guad_JHM2/'
# out_fld = '../output/San_Guad_JHM2/'
dir_path =  '../rapid_data_official/'
inp_fld = dir_path 
out_fld = dir_path

con_csv = inp_fld + 'rapid_connect_San_Guad.csv'
m3r_ncf = inp_fld + ('m3_riv_San_Guad_20100101_20131231_VIC0125_3H_utc_err_'
                     'R286_D.nc')
kpr_csv = inp_fld + 'k_San_Guad_2004_1.csv'
xpr_csv = inp_fld + 'x_San_Guad_2004_1.csv'

bas_csv = inp_fld + 'riv_bas_id_San_Guad_hydroseq.csv'

ZS_dtR = 900

Qou_ncf = out_fld + 'Qout_San_Guad_20100101_20131231_VIC0125_3H_utc.nc'


# *****************************************************************************
# Make sure files and folders exist
# *****************************************************************************
for fil_ext in [con_csv, kpr_csv, xpr_csv, m3r_ncf, bas_csv]:
    try:
        with open(fil_ext):
            pass
    except IOError:
        print('ERROR - Unable to open ' + fil_ext)
        raise SystemExit(22)

del fil_ext

os.makedirs(out_fld, exist_ok=True)


# *****************************************************************************
# Connectivity function
# *****************************************************************************
def con_vec(con_csv):

    IV_riv_tot = []
    IV_dwn_tot = []
    with open(con_csv, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            IV_riv_tot.append(int(row[0]))
            IV_dwn_tot.append(int(row[1]))
    IV_riv_tot = numpy.array(IV_riv_tot, dtype=numpy.int32)
    IV_dwn_tot = numpy.array(IV_dwn_tot, dtype=numpy.int32)

    return IV_riv_tot, IV_dwn_tot


# *****************************************************************************
# Basin function
# *****************************************************************************
def bas_vec(bas_csv):

    IV_riv_bas = []
    with open(bas_csv, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            IV_riv_bas.append(int(row[0]))
    IV_riv_bas = numpy.array(IV_riv_bas, dtype=numpy.int32)

    return IV_riv_bas


# *****************************************************************************
# Hash tables function
# *****************************************************************************
def hsh_tbl(IV_riv_tot, IV_riv_bas):

    IS_riv_tot = len(IV_riv_tot)
    IM_hsh_tot = {}
    for JS_riv_tot in range(IS_riv_tot):
        IM_hsh_tot[IV_riv_tot[JS_riv_tot]] = JS_riv_tot
    # IM_hsh_tot[IS_riv] = JS_riv_tot

    IS_riv_bas = len(IV_riv_bas)
    IM_hsh_bas = {}
    for JS_riv_bas in range(IS_riv_bas):
        IM_hsh_bas[IV_riv_bas[JS_riv_bas]] = JS_riv_bas
    # IM_hsh_bas[IS_riv] = JS_riv_bas

    IV_bas_tot = [IM_hsh_tot[IS_riv] for IS_riv in IV_riv_bas]
    IV_bas_tot = numpy.array(IV_bas_tot, dtype=numpy.int32)
    # This array allows for index mapping such that IV_riv_tot[JS_riv_tot]
    #                                             = IV_riv_bas[JS_riv_bas]
    # IV_bas_tot[JS_riv_bas] = JS_riv_tot

    return IM_hsh_tot, IM_hsh_bas, IV_bas_tot


# *****************************************************************************
# Network matrix function
# *****************************************************************************
def net_mat(IV_dwn_tot, IM_hsh_tot, IV_riv_bas, IM_hsh_bas):

    IS_riv_bas = len(IV_riv_bas)
    IV_row = []
    IV_col = []
    ZV_val = []
    for JS_riv_bas in range(IS_riv_bas):
        JS_riv_tot = IM_hsh_tot[IV_riv_bas[JS_riv_bas]]
        IS_dwn = IV_dwn_tot[JS_riv_tot]
        if IS_dwn != 0 and IS_dwn in IM_hsh_bas:
            JS_riv_ba2 = IM_hsh_bas[IS_dwn]
            IV_row.append(JS_riv_ba2)
            IV_col.append(JS_riv_bas)
            ZV_val.append(1)

    ZM_Net = csc_matrix((ZV_val, (IV_row, IV_col)),
                        shape=(IS_riv_bas, IS_riv_bas),
                        dtype=numpy.int32,
                        )

    return ZM_Net


# *****************************************************************************
# Muskingum k and x function
# *****************************************************************************
def k_x_vec(kpr_csv, xpr_csv, IV_bas_tot):

    ZV_kpr_tot = []
    with open(kpr_csv, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            ZV_kpr_tot.append(float(row[0]))
    ZV_kpr_tot = numpy.array(ZV_kpr_tot, dtype=numpy.float64)
    ZV_kpr_bas = ZV_kpr_tot[IV_bas_tot]

    ZV_xpr_tot = []
    with open(xpr_csv, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            ZV_xpr_tot.append(float(row[0]))
    ZV_xpr_tot = numpy.array(ZV_xpr_tot, dtype=numpy.float64)
    ZV_xpr_bas = ZV_xpr_tot[IV_bas_tot]

    return ZV_kpr_bas, ZV_xpr_bas


# *****************************************************************************
# Muskingum C1, C2, C3 function
# *****************************************************************************
def ccc_mat(ZV_kpr_bas, ZV_xpr_bas, ZS_dtR):

    ZV_den = ZS_dtR/2 + ZV_kpr_bas * (1 - ZV_xpr_bas)

    ZV_C1m = ZS_dtR/2 - ZV_kpr_bas * ZV_xpr_bas
    ZV_C1m = ZV_C1m / ZV_den
    ZM_C1m = diags(ZV_C1m, format='csc', dtype=numpy.float64)

    ZV_C2m = ZS_dtR/2 + ZV_kpr_bas * ZV_xpr_bas
    ZV_C2m = ZV_C2m / ZV_den
    ZM_C2m = diags(ZV_C2m, format='csc', dtype=numpy.float64)

    ZV_C3m = - ZS_dtR/2 + ZV_kpr_bas * (1 - ZV_xpr_bas)
    ZV_C3m = ZV_C3m / ZV_den
    ZM_C3m = diags(ZV_C3m, format='csc', dtype=numpy.float64)

    return ZM_C1m, ZM_C2m, ZM_C3m


# *****************************************************************************
# Muskingum routing matrices
# *****************************************************************************
def rte_mat(ZM_Net, ZM_C1m, ZM_C2m, ZM_C3m):

    IS_riv_bas = ZM_Net.shape[0]
    ZM_Idt = identity(IS_riv_bas, format='csc', dtype=numpy.int32)

    ZM_Lin = ZM_Idt - ZM_C1m * ZM_Net
    ZM_Qex = ZM_C1m + ZM_C2m
    ZM_Qou = ZM_C3m + ZM_C2m * ZM_Net

    return ZM_Lin, ZM_Qex, ZM_Qou


# *****************************************************************************
# Metadata of external inflow
# *****************************************************************************
def m3r_mdt(m3r_ncf):

    f = netCDF4.Dataset(m3r_ncf, 'r')

    # -------------------------------------------------------------------------
    # Check dimensions exist
    # -------------------------------------------------------------------------
    if 'rivid' not in f.dimensions:
        print('ERROR - rivid dimension does not exist in ' + m3r_ncf)
        raise SystemExit(22)

    if 'time' not in f.dimensions:
        print('ERROR - time dimension does not exist in ' + m3r_ncf)
        raise SystemExit(22)

    if 'nv' not in f.dimensions:
        print('ERROR - nv dimension does not exist in ' + m3r_ncf)
        raise SystemExit(22)

    if len(f.dimensions['nv']) != 2:
        print('ERROR - nv dimension is not 2 ' + m3r_ncf)
        raise SystemExit(22)

    # -------------------------------------------------------------------------
    # Check variables exist
    # -------------------------------------------------------------------------
    if 'rivid' not in f.variables:
        print('ERROR - rivid variable does not exist in ' + m3r_ncf)
        raise SystemExit(22)

    if 'time' not in f.variables:
        print('ERROR - time variable does not exist in ' + m3r_ncf)
        raise SystemExit(22)

    if 'time_bnds' not in f.variables:
        print('ERROR - time_bnds variable does not exist in ' + m3r_ncf)
        raise SystemExit(22)

    if 'm3_riv' not in f.variables:
        print('ERROR - m3_riv variable does not exist in ' + m3r_ncf)
        raise SystemExit(22)

    # -------------------------------------------------------------------------
    # Retrieve variables
    # -------------------------------------------------------------------------
    IV_m3r_tot = f.variables['rivid']
    IV_m3r_tim = f.variables['time']
    IM_m3r_tim = f.variables['time_bnds']

    IS_m3r_tim = len(IV_m3r_tim)
    ZS_TaR = IM_m3r_tim[0, 1] - IM_m3r_tim[0, 0]
    # Using IM_m3r_tim rather than IV_m3r_tim which may have only one timestep

    return IV_m3r_tot, IV_m3r_tim, IM_m3r_tim, IS_m3r_tim, ZS_TaR


# *****************************************************************************
# Time step correspondance
# *****************************************************************************
def stp_cor(ZS_TaR, ZS_dtR):

    if round(ZS_TaR/ZS_dtR) == ZS_TaR/ZS_dtR:
        IS_dtR = round(ZS_TaR/ZS_dtR)
    else:
        print('ERROR - quotient of time steps is not an integer')
        raise SystemExit(22)

    return IS_dtR


# *****************************************************************************
# Check topology
# *****************************************************************************
def chk_top(IV_riv_bas, IM_hsh_bas, IV_riv_tot, IV_dwn_tot, IM_hsh_tot):

    # -------------------------------------------------------------------------
    # Check for missing connections upstream
    # -------------------------------------------------------------------------
    IS_riv_tot = len(IV_riv_tot)
    for JS_riv_tot in range(IS_riv_tot):
        IS_riv = IV_riv_tot[JS_riv_tot]
        IS_dwn = IV_dwn_tot[JS_riv_tot]
        if IS_dwn != 0:
            if IS_dwn in IM_hsh_bas and IS_riv not in IM_hsh_bas:
                print('WARNING - connectivity: ' + str(IS_riv) +
                      ' is upstream of ' + str(IS_dwn) +
                      ' but is not the basin file')

    # -------------------------------------------------------------------------
    # Check for missing connections downstream
    # -------------------------------------------------------------------------
    for IS_riv in IV_riv_bas:
        IS_dwn = IV_dwn_tot[IM_hsh_tot[IS_riv]]
        if IS_dwn != 0:
            if IS_dwn not in IM_hsh_bas:
                print('WARNING - connectivity: ' + str(IS_dwn) +
                      ' is downstream of ' + str(IS_riv) +
                      ' but is not the basin file')

    # -------------------------------------------------------------------------
    # Check sorting from upstream to downstream
    # -------------------------------------------------------------------------
    for IS_riv in IV_riv_bas:
        IS_dwn = IV_dwn_tot[IM_hsh_tot[IS_riv]]
        if IS_dwn != 0:
            if IS_dwn in IM_hsh_bas:
                if IM_hsh_bas[IS_dwn] < IM_hsh_bas[IS_riv]:
                    print('ERROR - sorting problem: ' + str(IS_dwn) +
                          ' is downstream of ' + str(IS_riv) +
                          ' but is located above in basin file')
                    raise SystemExit(22)


# *****************************************************************************
# Check IDs
# *****************************************************************************
def chk_ids(IV_riv_tot, IV_m3r_tot):

    if not numpy.all(IV_riv_tot - IV_m3r_tot == 0):
        print('ERROR - The river IDs in con_csv and m3r_ncf differ')
        raise SystemExit(22)


# *****************************************************************************
# Check time variables
# *****************************************************************************
def chk_tim(IV_m3r_tim, IM_m3r_tim, ZS_TaR):

    if len(IV_m3r_tim) != len(IM_m3r_tim):
        print('ERROR - The time and time_bnds variables have different sizes')
        raise SystemExit(22)

    if not numpy.all(IV_m3r_tim - IM_m3r_tim[:, 0] == 0):
        print('ERROR - inconsistent values in time and time_bnds[0]')
        raise SystemExit(22)

    if not numpy.all(IV_m3r_tim - IM_m3r_tim[:, 1] == -ZS_TaR):
        print('ERROR - inconsistent values in time and time_bnds[1]')
        raise SystemExit(22)

    if not numpy.all(IV_m3r_tim[:-1] - IV_m3r_tim[1:] == -ZS_TaR):
        print('ERROR - uneven increment in time ')
        raise SystemExit(22)


# *****************************************************************************
# Metadata of outflow
# *****************************************************************************
def Qou_mdt(m3r_ncf, Qou_ncf, IV_bas_tot):

    # -------------------------------------------------------------------------
    # Get UTC date and time
    # -------------------------------------------------------------------------
    YS_dat = datetime.now(timezone.utc)
    YS_dat = YS_dat.replace(microsecond=0)
    YS_dat = YS_dat.isoformat()+'+00:00'

    # -------------------------------------------------------------------------
    # Open one file and create the other
    # -------------------------------------------------------------------------
    f = netCDF4.Dataset(m3r_ncf, 'r')
    g = netCDF4.Dataset(Qou_ncf, 'w', format='NETCDF4')

    # -------------------------------------------------------------------------
    # Copy dimensions
    # -------------------------------------------------------------------------
    YV_exc = ['nerr']
    for nam, dim in f.dimensions.items():
        if nam not in YV_exc:
            g.createDimension(nam, len(dim) if not dim.isunlimited() else None)

    g.createDimension('nerr', 3)

    # -------------------------------------------------------------------------
    # Create variables
    # -------------------------------------------------------------------------
    g.createVariable('Qout', 'float32', ('time', 'rivid'))
    g['Qout'].long_name = ('average river water discharge downstream of '
                           'each river reach')
    g['Qout'].units = 'm3 s-1'
    g['Qout'].coordinates = 'lon lat'
    g['Qout'].grid_mapping = 'crs'
    g['Qout'].cell_methods = 'time: mean'

    g.createVariable('Qout_err', 'float32', ('nerr', 'rivid'))
    g['Qout_err'].long_name = ('average river water discharge uncertainty '
                               'downstream of each river reach')
    g['Qout_err'].units = 'm3 s-1'
    g['Qout_err'].coordinates = 'lon lat'
    g['Qout_err'].grid_mapping = 'crs'
    g['Qout_err'].cell_methods = 'time: mean'

    # -------------------------------------------------------------------------
    # Copy variables variables
    # -------------------------------------------------------------------------
    YV_exc = ['m3_riv', 'm3_riv_err']
    YV_sub = ['rivid', 'lon', 'lat']
    for nam, var in f.variables.items():
        if nam not in YV_exc:
            if nam in YV_sub:
                g.createVariable(nam, var.datatype, var.dimensions)
                g[nam][:] = f[nam][IV_bas_tot]

            else:
                g.createVariable(nam, var.datatype, var.dimensions)
                g[nam][:] = f[nam][:]

            g[nam].setncatts(f[nam].__dict__)
            # copy variable attributes all at once via dictionary

    # -------------------------------------------------------------------------
    # Populate global attributes
    # -------------------------------------------------------------------------
    g.Conventions = f.Conventions
    g.title = f.title
    g.institution = f.institution
    g.source = 'RAPID, ' + 'runoff: ' + os.path.basename(m3r_ncf)
    g.history = 'date created: ' + YS_dat
    g.references = ('https://doi.org/10.1175/2011JHM1345.1, '
                    'https://github.com/c-h-david/rapid')
    g.comment = ''
    g.featureType = f.featureType

    # -------------------------------------------------------------------------
    # Close all files
    # -------------------------------------------------------------------------
    f.close()
    g.close()


# *****************************************************************************
# Muskingum routing
# *****************************************************************************
def mus_rte(ZM_Lin, ZM_Qex, ZM_Qou, IS_dtR, ZV_Qou_ini, ZV_Qex_avg):

    ZV_Qou = ZV_Qou_ini
    ZV_avg = numpy.zeros(len(ZV_Qou_ini))
    ZV_rh1 = ZM_Qex * ZV_Qex_avg

    for JS_dtR in range(IS_dtR):
        # ---------------------------------------------------------------------
        # Updating average before routing to remain in [0, IS_dtR - 1] range
        # ---------------------------------------------------------------------
        ZV_avg = ZV_avg + ZV_Qou

        # ---------------------------------------------------------------------
        # Updating instantaneous value of right-hand side
        # ---------------------------------------------------------------------
        ZV_rhs = ZV_rh1 + ZM_Qou * ZV_Qou

        # ---------------------------------------------------------------------
        # Routing
        # ---------------------------------------------------------------------
        # ZV_Qou = spsolve(ZM_Lin, ZV_rhs)
        ZV_Qou = spsolve_triangular(ZM_Lin, ZV_rhs,
                                    lower=True, unit_diagonal=True)
        # ZV_Qou = slv_fac(ZV_rhs)
        # ZV_Qou = slv_umf.solve(ZV_rhs)
        # ZV_Qou = slv_slu.solve(ZV_rhs)

    ZV_avg = ZV_avg / IS_dtR

    ZV_Qou_avg = ZV_avg
    ZV_Qou_fin = ZV_Qou

    return ZV_Qou_avg, ZV_Qou_fin


# *****************************************************************************
# Initialize
# *****************************************************************************

# -----------------------------------------------------------------------------
# River network
# -----------------------------------------------------------------------------
IV_riv_tot, IV_dwn_tot = con_vec(con_csv)
IV_riv_bas = bas_vec(bas_csv)
IM_hsh_tot, IM_hsh_bas, IV_bas_tot = hsh_tbl(IV_riv_tot, IV_riv_bas)
ZM_Net = net_mat(IV_dwn_tot, IM_hsh_tot, IV_riv_bas, IM_hsh_bas)

# -----------------------------------------------------------------------------
# Model parameters
# -----------------------------------------------------------------------------
ZV_kpr_bas, ZV_xpr_bas = k_x_vec(kpr_csv, xpr_csv, IV_bas_tot)
ZM_C1m, ZM_C2m, ZM_C3m = ccc_mat(ZV_kpr_bas, ZV_xpr_bas, ZS_dtR)
ZM_Lin, ZM_Qex, ZM_Qou = rte_mat(ZM_Net, ZM_C1m, ZM_C2m, ZM_C3m)

# -----------------------------------------------------------------------------
# Metadata of external inflow, and time step correspondance
# -----------------------------------------------------------------------------
IV_m3r_tot, IV_m3r_tim, IM_m3r_tim, IS_m3r_tim, ZS_TaR = m3r_mdt(m3r_ncf)
IS_dtR = stp_cor(ZS_TaR, ZS_dtR)

# -----------------------------------------------------------------------------
# Initialize discharge
# -----------------------------------------------------------------------------
ZV_Qou_ini = numpy.zeros(len(IV_riv_bas), dtype=numpy.float64)

# -----------------------------------------------------------------------------
# General checks
# -----------------------------------------------------------------------------
chk_top(IV_riv_bas, IM_hsh_bas, IV_riv_tot, IV_dwn_tot, IM_hsh_tot)
chk_ids(IV_riv_tot, IV_m3r_tot)
chk_tim(IV_m3r_tim, IM_m3r_tim, ZS_TaR)

# -----------------------------------------------------------------------------
# Metadata of outflow
# -----------------------------------------------------------------------------
Qou_mdt(m3r_ncf, Qou_ncf, IV_bas_tot)


# *****************************************************************************
# Routing
# *****************************************************************************
slv_fac = factorized(ZM_Lin)
slv_slu = splu(ZM_Lin)
# slv_umf = umlu(ZM_Lin)

f = netCDF4.Dataset(m3r_ncf, 'r')
g = netCDF4.Dataset(Qou_ncf, 'a')
Qout = g.variables['Qout']
discharge_estimation = []
divider = 8  

for JS_m3r_tim in range(IS_m3r_tim//divider):
    Z_ave_day = ZV_Qou_ini
    time1 = time.time()
    for i in range(divider):
        # ZV_Qex_avg = f.variables['m3_riv'][JS_m3r_tim][IV_bas_tot] / ZS_TaR
        ZV_Qex_avg = f.variables['m3_riv'][JS_m3r_tim*divider+i][IV_bas_tot] / ZS_TaR

        ZV_Qou_avg, ZV_Qou_fin = mus_rte(ZM_Lin, ZM_Qex, ZM_Qou, IS_dtR,
                                        ZV_Qou_ini, ZV_Qex_avg)
        ZV_Qou_ini = ZV_Qou_fin

        Qout[JS_m3r_tim, :] = ZV_Qou_avg[:]

        Z_ave_day += ZV_Qou_avg
            
    discharge_estimation.append(Z_ave_day/divider)
    time2 = time.time()
    
    print(f"day: {JS_m3r_tim} time: {time2-time1}")
    
    m3_riv_df = pd.DataFrame(ZM_Qex[0:100,0:100])
    m3_riv_df.to_csv('../model_saved_official_original/ZM_Qex_official.csv', index=False)
    m3_riv_df = pd.DataFrame(ZM_Qou[0:100,0:100])
    m3_riv_df.to_csv('../model_saved_official_original/ZM_Qou_official.csv', index=False)
    m3_riv_df = pd.DataFrame(ZM_Lin[0:100,0:100])
    m3_riv_df.to_csv('../model_saved_official_original/ZM_Lin_official.csv', index=False)
    
m3_riv_df = pd.DataFrame(ZM_Net.todense())
m3_riv_df.to_csv('../model_saved_official_original/ZM_Net.csv', index=False)
numpy.savetxt("../model_saved_official_original/discharge_est_offical.csv", discharge_estimation, delimiter=",")
m3_riv_df = pd.DataFrame(ZM_C1m)
m3_riv_df.to_csv('../model_saved_official_original/ZM_C1m.csv', index=False)
m3_riv_df = pd.DataFrame(ZM_C2m)
m3_riv_df.to_csv('../model_saved_official_original/ZM_C2m.csv', index=False)
m3_riv_df = pd.DataFrame(ZM_C3m)
m3_riv_df.to_csv('../model_saved_official_original/ZM_C3m.csv', index=False)
Qout_df = pd.DataFrame(Qout[:])
Qout_df.to_csv('../model_saved_official_original/Qout.csv', index=False)
m3_riv_df = pd.DataFrame(f.variables['m3_riv'][:])
m3_riv_df.to_csv('../model_saved_official_original/m3_riv_df.csv', index=False)
f.close()
g.close()


# *****************************************************************************
# Diagnosis
# *****************************************************************************
# print(IS_m3r_tim)
# print(ZS_TaR)
# print(ZS_dtR)
# print(ZS_TaR/ZS_dtR)
# print(dir())
# print(ZM_Net.todense())
# print('-----')
# print(ZM_C1m)
# print('-----')
# print(ZM_C2m)
# print('-----')
# print(ZM_C3m)
# print('-----')
# print(ZM_ICN)
# print([name for name in globals() if not name.startswith('_')])


# *****************************************************************************
# End
# *****************************************************************************
