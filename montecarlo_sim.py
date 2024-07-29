# THIS SCRIPT PERFORMS MONTE CARLO SIMULATIONS TO CALCULATE HOW LIKELY IT IS THAT BINARY MOTION WOULD CREATE A LARGER DELTA_RV THAN WHAT IS OBSERVED
# (DELTA_RV is the difference between the highest and the lowest radial velocity measurement) 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from astropy.io import ascii
from scipy.optimize import curve_fit
import time
from matplotlib import rcParams
rcParams['font.family'] = 'STIXGeneral'
c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c10b = '#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf', '#a34f88'
cs = [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c10b]
start = time.time()

save_fig = False

# ASSIGN SOME VALUES THAT WE WILL USE LATER
whose_data = 's23' #f03, k22, s23  # foellmi+2003, KaranDsilva+2022, Schootemeijer+2024
s_id       = 9  # 1 --> SMC AB1 etc
mwr     = 20  # assumed mass of the Wolf-Rayet star, in Solar units
pdist   = 'powerlaw' # Vasco, Langer, Renzo, or powerlaw. Papers: Langer+2020, Renzo+2019
logp_exponent = 0. # if we choose powerlaw.  #1.44 is pi_max lmc, -0.18 is pi_max smc based on obs periods
logpmin, logpmax = 0.00001, 3.5

eccdist = 'exponential'#'exponential'   # circular, exponential     # which eccentricity distribution?
ecc_exp_factor= 0    # if we choose exponential
qmin, qmax = 0.05, 2 # min. and max. mass ratio of the hypothetical binary components
n_universes   = 11 # 1 universe mean: simulate once per P_orb+q combination
sigma_line_var  = 0      # non_binary variability. Enter 0 if we do not want to bother. units: km/s
set_err_to_zero = True

# let save name of figure depend on assumptions about eccentricity etc
suf = 'circ_errv%s'%sigma_line_var
if set_err_to_zero == False: suf = 'circ_errv%s_ccferr'%sigma_line_var
if eccdist == 'exponential': suf = 'flatecc'
save_as = 'figs/AB%s_%s_%s.png'%( s_id,whose_data,suf )

# make a function to randomly draw orbital parameters from
def draw_orbital_conf(fbin):
    a, b, c, d = np.random.random(), np.random.random(), np.random.random(), np.random.random()
    # do we draw a binary?
    binary, p_in_d, rat, poffset, omega, ecc = 0, 0, 0, 0, 0, 0
    if a < fbin: binary = 1
    if binary == 1:
        logp = np.random.choice(list(logps), p=list(logp_probs))
        p_in_d  = 10**logp
        rat     = qmin + (qmax-qmin)*b       
        poffset = c*1.e7                 # random period offset
        # eccentricy, if it is taken into account
        if not eccdist == 'circular':
            omega   = d*2*np.pi               # random orientation of this longitude of ascending note or whatever we call this
            ecc     = np.random.choice( eccs, p=ecc_probs )
    return binary, p_in_d, rat, poffset, omega, ecc

# draw the orbital parameters from the lists that we define below
# make lists with inclination
incs, incprobs     = np.linspace(0, np.pi/2, 2701), []
for inc in incs:
    incprobs.append(np.sin(inc))
incprobs = incprobs/sum(incprobs)
# make lists with logps 
logps_pl, logp_probs_pl     = np.linspace(logpmin, logpmax,335), []
for logp_pl in logps_pl:  #pl as in power law, one that isnt broken
    logp_probs_pl.append( logp_pl**logp_exponent )
logp_probs_pl = logp_probs_pl/sum(logp_probs_pl)
# make lists with eccentricities
eccs, ecc_probs     = np.linspace(0, 0.9, 451), []
for ecc in eccs:  #pl as in power law, one that isnt broken
    ecc_probs.append( np.exp( ecc_exp_factor*ecc ) )
ecc_probs = ecc_probs/sum(ecc_probs)

# here we define a extra orbital period distributions that can be also be used
# broken power law Porb distribution
if pdist == 'Vasco':
    x1, x2, x3 = 0., 1.25, 3.5   # x2 is peak, others are where y=0. Say peak is at y=1 (we norm. ltr)
if pdist == 'Langer':
    x1, x2, x3 = 0.5, 2.25, 3    # x2 is peak, others are where y=0. Say peak is at y=1 (we norm. ltr)
if pdist == 'Vasco' or pdist == 'Langer':
    y1, y2, y3 = 0., 1., 0.
    slope12 = (y2-y1)/(x2-x1)
    cept12  = y1 - slope12*x1
    slope23 = (y3-y2)/(x3-x2)
    cept23  = y2 - slope23*x2
    print(( slope12, cept12, slope23, cept23 ))
    x,y = np.linspace(x1,x3,101), []
    y12 = x*slope12+cept12
    y23 = x*slope23+cept23
    for f in range(len(x)):
        y.append( min(y12[f],y23[f]) )
    logp_probs_bpl, logps_bpl = y/sum(y), x
if pdist == 'powerlaw':
    logp_probs , logps = logp_probs_pl, logps_pl
if pdist == 'Vasco' or pdist == 'Langer':
    logp_probs , logps = logp_probs_bpl, logps_bpl

# with the function below we can get the radial velocity semi-amplitude of a certain binary
grav_const, msol = 6.67e-11, 1.99e30 #si
def get_k( p_in_d, rat, inc, ecc ):   # eq 2.92 in M. Benaquista book
    porb = p_in_d * 86400  # go to porb in s
    m2 = mwr * rat
    mtot = mwr + m2
    kwr = ( (m2*msol)**3/(mtot*msol)**2 * (grav_const*2*np.pi)/(porb) )**(1./3) * 0.001
    kwr = kwr * np.sin(inc)
    kwr = kwr / np.sqrt( 1-ecc*ecc ) # account for eccentricity
    return kwr

# with the function below we can get the radial velocity of a certain binary at a certain moment
def get_rv_at_a_moment_in_time( kwr,tobs_i,p_in_d,poffset ):
    tobs_i_mod = (tobs_i+poffset)%p_in_d
    phase = 2*np.pi * (tobs_i_mod/p_in_d)
    rv_i = kwr * np.sin(phase)
    return rv_i

def get_phase( tobs_i,p_in_d,poffset ):
    tobs_i_mod = (tobs_i+poffset)%p_in_d
    phase = tobs_i_mod/p_in_d
    return phase

# this is to also include eccentricity:
from math import sin
from scipy.optimize import fsolve
def func(y, val):
    x = ( y-ecc*sin(y) ) / (2.*np.pi) - val
    return x
def get_RV( K,omega,gamma,ecc, phase ):
    E_min_sinE  = phase                # phase is equivalent to random E_min_sinE norm. to 1 but I fear I might forget 
    lala        = fsolve( func,0.5,E_min_sinE )
    E           = lala[0]
    if E <= np.pi:
        theta = np.arccos(  (np.cos(E)-ecc) / (1 - ecc*np.cos(E))  )
    if E > np.pi:   # arccos returns only values between 0 and pi, theta goes to 2+pi in real life
        theta = 2*np.pi - np.arccos(  (np.cos(E)-ecc) / (1 - ecc*np.cos(E))  )  # tested that it is symmetric
    RV = K * ( np.cos(theta+omega)+ecc*np.cos(omega) ) + gamma
    return RV

# prepare for plotting 
plt.figure(figsize=(7, 14/3.), dpi=200 )
ax=plt.subplot(111 )
# here we make lists of lists because we call the lists later when we loop over smc and lmc (although usually we only enter 1 s_id etc)
s_ids = [ s_id ]
m_ids = [1,2,3,6,7,10,110,127,128,133,138,139,141,151,152,157]
m_ids = [1] # karan+22 fig 5 is only WR1
ids_list = [s_ids]

rv_file_m = np.empty(( n_universes,len(m_ids) ))
rv_files  = [ rv_file_m ] 
p_file    = np.empty(( n_universes,len(m_ids) ))

# which RV measurements do we consider?
if whose_data == 'f03': 
    tcol, rvcol,rverrcol = 'x','Curve1','err_vis'
    fldr = 'data'
if whose_data == 'k22': 
    tcol, rvcol,rverrcol = 'bjd','rv','sigma_rv'
    fldr = 'data_karan'
if whose_data == 's23': 
    tcol, rverrcol = 'mjd','sigma_rv'
    fldr = 'data_ourobs'

# define the P_orb and mass ratio parameter space that we explore
nlogps, nrats = 36,40     # Dsilva22: 51,39
logps_sim = np.linspace( logpmin,logpmax,nlogps )
rats_sim  = np.linspace( qmin,qmax,nrats )
X,Y       = np.meshgrid( logps_sim,rats_sim*20 )
Z         = np.zeros( (nrats,nlogps) )


# NOW THAT WE HAVE DEFINED ALL FUNCTIONS ETC, we can start the Monte Carlo simulations
# TOTAL nr of simulated binaries: n_logps * n_rats * n_universes ( nr of periods * nr of mass ratios * simulations per combination )
clist = ['#000000','#ffdd00','#ff8800','#ff2200']   # list of colors for the scatter points
f_detected_list_rat_integrated, f_detected_specific_rat = [], []
for g in [0]:    # hypothetically this indent could be removed
    rv_file = rv_files[g]
    ids = ids_list[g]
    deltarv_obs = []
    if whose_data == 's23': rvcol = 'rv'
    for i,id_i in enumerate(ids):  # indent also not necessary anymore because we stopped running the script for multiple sources
        if whose_data == 'f03': s = ascii.read( '%s/ab%s.csv'%(fldr,id_i) )# ab for f03; AB for s23
        if whose_data == 's23': s = ascii.read( '%s/AB%s.csv'%(fldr,id_i) )# ab for f03; AB for s23
        tobs  = s[tcol]
        vrads = s[rvcol]
        vrad_errs = s[rverrcol]
        print('ID %s'%id_i)
        delta_rv_tresh = max(vrads) - min(vrads)
        for x,logp in enumerate(logps_sim):     # get period
            p_in_d = 10**logp
            delta_rv_list_rat_integrated = []
            for y,rat in enumerate(rats_sim):   # get mass ratio
                delta_rv_list = []
                for j in range( n_universes):   # for n_universes times, we draw a binary star constellation
                    binary_notused,p_in_d_notused,rat_notused,poffset, omega, ecc = draw_orbital_conf(1) # we don't use binary because we assume a binary fraction of 1
                    inc = np.random.choice(incs, p=incprobs)    # draw inclination based on lists that we made
                    kwr = get_k(p_in_d, rat, inc, ecc)          # get radial velocity amplitude
                    klist = []
                    for k in range( len( tobs ) ):              # make a list of radial velocity measurements in the synthetic binary
                        if set_err_to_zero == False:
                            rand_err = np.random.normal(0, vrad_errs[k] )
                        else: rand_err = 0
                        line_var_err = 0
                        if sigma_line_var > 0:
                            line_var_err = np.random.normal( 0,sigma_line_var )
                        phase = get_phase( tobs[k],p_in_d,poffset )
                        if ecc == 0:
                            rv_k  = kwr * np.cos( 2*np.pi*phase )
                        elif ecc > 0:
                            rv_k  = get_RV( kwr,omega,0,ecc, phase )   # 0 is gamma, we don't care about v_sys
                        klist.append( rv_k + rand_err +line_var_err)
                    delta_rv = max(klist) - min(klist)
                    delta_rv_list.append(delta_rv)
                    delta_rv_list_rat_integrated.append(delta_rv)
                delta_rv_list = np.array(delta_rv_list)
                delta_rv_u50_list = delta_rv_list[ delta_rv_list < delta_rv_tresh ]
                f_detected = 1-float(len(delta_rv_u50_list)) / len(delta_rv_list)
                print('x=%s  y=%s  fraction detected:'%(x,y), np.round( f_detected,2) )
                Z[y,x] = f_detected
                color = clist[0]
                if f_detected < 0.95: color=clist[1] 
                if f_detected < 0.50: color=clist[2]
                if f_detected < 0.05: color=clist[3]
                marker, mas = 'x', 16
                if whose_data == 'f03': marker,mas = 'o', 10
                if rat == 0.25 and p_in_d < 365.25:
                    f_detected_specific_rat.append( f_detected )
                ax.scatter( logp,rat*20,s=mas,c=color,lw=1.5, marker=marker )
            delta_rv_list_rat_integrated = np.array( delta_rv_list_rat_integrated )
            delta_rv_u50_list_rat_integrated = delta_rv_list_rat_integrated[ delta_rv_list_rat_integrated < delta_rv_tresh ]
            f_detected_rat_integrated = 1-float(len(delta_rv_u50_list_rat_integrated)) / len(delta_rv_list_rat_integrated)
            f_detected_list_rat_integrated.append( f_detected_rat_integrated )
            print('		f_detected_list_rat_integrated', f_detected_list_rat_integrated)

print("	5M and Porb< 1yr:", f_detected_specific_rat )
print("	average for 5M and Porb< 1yr:", np.average(f_detected_specific_rat) )

labellist = ['$p(\Delta \mathrm{RV}_\mathrm{sim} > \Delta \mathrm{RV}_\mathrm{obs}) > 0.95$', '$0.95 > p(\Delta \mathrm{RV}_\mathrm{sim} > \Delta \mathrm{RV}_\mathrm{obs}) > 0.5$', '$0.5 > p(\Delta \mathrm{RV}_\mathrm{sim} > \Delta \mathrm{RV}_\mathrm{obs}) > 0.05$', '$p(\Delta \mathrm{RV}_\mathrm{sim} > \Delta \mathrm{RV}_\mathrm{obs}) < 0.05$']
for i,c in enumerate(clist):
	ax.scatter( 0, -5,  s=20,c=c,lw=1.5, marker='x', label=labellist[i] )
loco = 'lower right'
if whose_data == 's23': loco = 'upper left'
ax.legend( loc=loco, framealpha=0.93, prop={'size':10} )
prefix = ''
if whose_data == 'f03': prefix = 'Foellmi+2003 campaign. '
suffix = ''
if eccdist == 'exponential': suffix = 'Flat ecc. distribution.'
if set_err_to_zero == True:
	ax.set_title('%sSMC AB%s. %s$\Delta$RV$_\mathrm{obs}$ = %s$\,$km/s'%(prefix, id_i, suffix, int(delta_rv_tresh+0.5) ), size=13 )
if set_err_to_zero == False:
	ax.set_title('%sSMC AB%s. %s$\Delta$RV$_\mathrm{obs}$ = %s$\,$km/s. $\\sigma_\mathrm{RV, \,mock} = %s\,$km/s'%(prefix, id_i, suffix, int(delta_rv_tresh+0.5), sigma_line_var ), size=13 )
# 1 yr line
ax.axvline( np.log10(362.25), lw=2, ls='--',color='#4da7c9' )

###################################
# we are basically done but we still would like to calculate a few detection probabilities for different orbital periods and mass ratio distributions
# make lists with Langer+2020 probs
pprobs_l20, qprobs_l20 = np.zeros(36), np.zeros(40)
for i,p in enumerate( np.linspace( 0, 3.5, 36) ):
	if p<0.45 or p>3.05: continue
	if p<2.3: pprobs_l20[i] = 4*p/7.  - 2/7.
	if p>2.3: pprobs_l20[i] = -4*p/3. + 4
for i,q in enumerate( np.linspace( 0.05, 2, 40) ):
	if q<0.325 or q>1.625: continue
	if q<0.625: qprobs_l20[i] = 10*q/3. - 1
	if q>0.625: qprobs_l20[i] = -q      + 1.6

# l20 dists
prob_l20, prob_total_l20 = 0,0
prob_l20_30d_plus, prob_total_l20_30d_plus = 0,0
for i, row in enumerate(Z):
	for j in range(len(row)):
		print( i, rats_sim[i], j, logps_sim[j], row[j])
		pprob_j ,qprob_i = pprobs_l20[j], qprobs_l20[i]
		prob_total_l20 += pprob_j*qprob_i
		prob_l20 += row[j] * pprob_j*qprob_i   # row[j] is p_observe
		if logps_sim[j] < 1.5: pprob_j = 0    # EXPERIMENT: take them out if the distribution because they have been detected
		prob_total_l20_30d_plus += pprob_j*qprob_i
		prob_l20_30d_plus       += row[j] * pprob_j*qprob_i   # row[j] is p_observe
print( "\nLanger+20 logPdist qdist integrated prob: %s"%(prob_l20/prob_total_l20) )
print( "Langer+20 logPdist 30d+ qdist integrated prob: %s"%(prob_l20_30d_plus/prob_total_l20_30d_plus) )

# Renzo+2019 dists
r19        = ascii.read( 'ramsj/pdist_renzo19.csv' )
pprobs_r19 = r19['prob']
prob_r19, prob_total_r19 = 0,0
prob_r19_30d_plus, prob_total_r19_30d_plus = 0,0
for i, row in enumerate(Z):
	for j in range(len(row)):
		pprob_j ,qprob_i = pprobs_r19[j], 0
		if rats_sim[i] >= 1.5: qprob_i = 1
		prob_total_r19 += pprob_j*qprob_i
		prob_r19       += row[j] * pprob_j*qprob_i  # row[j] is p_observe
		if logps_sim[j] < 1.5: pprob_j = 0    # EXPERIMENT: take them out if the distribution because they have been detected
		prob_total_r19_30d_plus += pprob_j*qprob_i
		prob_r19_30d_plus       += row[j] * pprob_j*qprob_i  # row[j] is p_observe
print( "Renzo+19 logPdist qdist integrated prob: %s"%(prob_r19/prob_total_r19) )
print( "Renzo+19 logPdist 30d+ qdist integrated prob: %s"%(prob_r19_30d_plus/prob_total_r19_30d_plus) )

# flat pdists
prob, prob_1yr, prob_qmin, prob_1yr_qmin, prob_30dplus= 0, 0, 0, 0, 0
for i, row in enumerate(Z):
	prob += sum(row)
	if rats_sim[i] > 0.145*2:  # initial mass ration was twice as large; Mini = 40, Mcurr = 20. So we proceed if M>5Msol or M>=6Msol, however you call it
		prob_qmin += sum(row)
	for j in range(len(row)):
		if logps_sim[j] < np.log10(365.25):
			prob_1yr += row[j]
			if rats_sim[i] > 0.145*2: 
				prob_1yr_qmin += row[j]
		if logps_sim[j] > np.log10(30):
			prob_30dplus += row[j]
print('prob/tot', prob/(nlogps*nrats) )
print('prob_qmin/tot', prob_qmin/(nlogps*(nrats-5) ))  # last 10 logP bins are larger than 1yr
print('prob_1yrmax/tot', prob_1yr/((nlogps-10)*nrats) )   # last 10 logP bins are larger than 1yr
print('prob_1yrmax_qmin/tot', prob_1yr_qmin/((nlogps-10)*(nrats-5)) )   # last 10 logP bins are larger than 1yr
print('prob_30dplus/tot', prob_30dplus/((nlogps-15)*(nrats)) )   # last 10 logP bins are larger than 1yr

# draw contours..
ax.contour( X,Y,Z,[0.05, 0.5, 0.95],linewidth=0.4, colors=['k'] )#,cmap='autumn' )
# .. and finally name the axes etc.
print('We have used:   Delta_RV,obs =',delta_rv_tresh)
print('that took %s sec.'%(int(time.time()-start+0.5)))
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
ax.tick_params( axis='both',direction='in')
#plt.tight_layout()
ax.set_xlim(-0.01,3.59)
ax.set_ylim(0,40.5)
ax.set_xlabel('$\log (P_\mathrm{orb}/\mathrm{d})$', size=12)
ax.set_ylabel('Companion mass [M$_\odot$]', size=12)
if save_fig == True:
	suf = 'circ'
	if eccdist == 'exponential': suf = 'flatecc'
	print('SAVED THAT')
	plt.savefig(save_as)#'figs/ab%s_%s_%s.pdf'%( s_ids[0],whose_data,suf ) )
plt.show()
