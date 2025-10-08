import os
import re
import math
import simpy
import random
import itertools
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
os.chdir('C:/Users/obriene/Projects/Discrete Event Simulation/MSDEC model')
from uhpt_population_projection import age_bands

################################################################################
                                  ####MODEL####
################################################################################

class default_params():
    ##############################USER PARAMETERS###############################
    scenario_name = 'MSDEC new build'
    #####Time between ococupancy samples
    occ_sample_time = 60
    #####run times and iterations
    start_year = datetime.today().year
    run_years = 10
    run_time = run_years * (365 * (60*24))
    iterations = 100
    #####resources
    no_chairs = 10000
    #####empty list for results
    pat_res = []
    occ_res = []
    #####Open and Close times and min/max LoS
    MSDEC_open = 8
    MSDEC_close = 22
    max_LoS = (MSDEC_close - MSDEC_open) * 60
    min_LoS = 120

    #############################DYNAMIC PARAMETERS#############################
    #####Current demand
    #Use the last year of data from end of previous month to today
    start = '31-05-2024'
    end = '31-05-2025'
    sdmart_engine = create_engine('mssql+pyodbc://@SDMartDataLive2/InfoDB?'\
                            'trusted_connection=yes&driver=ODBC+Driver+17'\
                                '+for+SQL+Server')
    msdec_query = f"""SET NOCOUNT ON;
    SET ANSI_WARNINGS OFF;

    --base data
    select	refdt.FinancialYear
            ,refdt.[MonthName]
            ,refdt.FinancialMonthNumber
            ,ipact.prcae_refno
            ,ipact.prvsp_refno
            ,ipact.admit_dttm
            ,ipact.disch_dttm
            ,ipact.fce_start_ward
            ,ipact.fce_end_ward
            ,ipact.fce_start_site
            ,ipact.fce_end_site
            ,spec.pfmgt_spec
            ,ipact.patcl
            ,ipact.proca_refno
            ,ipact.spell_los  --SP 1.1 22/8/2023
            ,ipact.pat_age_on_admit
    into 	#ip 	--drop table #ip
    from InfoDB.dbo.vw_ipdc_fces_pfmgt ipact
    left join infodb.dbo.vw_cset_specialties spec		on ipact.local_spec = spec.local_spec
    left join PiMSMarts.Reference.[Date] refdt			on CAST(ipact.disch_dttm as date) = refdt.[Date]
    where  ipact.disch_dttm between '{start}' and '{end} 23:59:59'  -- getdate()
    and ipact.last_episode_in_spell = 1
    and ipact.admet between '21' and '83' --not in ('11','12','13')
    and Spec.nat_spec not in ('199','223','290','291','331','344','345','346','360','424','499','501','560'
                            ,'650','651','652','653','654','655','656','657','658','659','660','661','662'
                            ,'700','710','711','712','713','715','720','721','722','723','724','725','726'
                            ,'727','730','840','920') --exclude non acute specialties
    and (ipact.AAUSpell not in ('L','O') or ipact.AAUSpell is null)
    and ipact.adcat = '01' --NHS Patients
    --and (ipact.spell_los > 0 or ipact.spell_los  is null)
    and fce_start_ward = 'RK950AAU'


    ---Join to ipmovements to get time on each ward. More rows in this bit because some patients have 2 stays on MSDEC within their spell							
    select  mov.sstay_start_dttm as MSDECStartDateTime
            ,mov.sstay_end_dttm as MSDECEndDateTime
            ,#ip.pat_age_on_admit as PatientAge
    from #ip  
    left join pimsmarts.dbo.ip_movements mov on mov.prvsp_refno = #ip.prvsp_refno
    where move_reason_sp = 'S'
    and sstay_ward_code = 'RK950AAU'"""
    current_msdec = pd.read_sql(msdec_query, sdmart_engine)
    current_msdec = current_msdec.loc[(current_msdec['MSDECStartDateTime']
                                       > pd.to_datetime(start, dayfirst=True))
                                    & (current_msdec['MSDECStartDateTime']
                                       < pd.to_datetime(end, dayfirst=True))
                                       ].copy()
    
    current_msdec['Weekend'] = np.where(current_msdec['MSDECStartDateTime'].dt
                                    .day_name().isin(['Saturday', 'Sunday']),
                                    'Weekend', 'Weekday')
    #Calculate LoS, filter out any >12h
    current_msdec['LoS'] = ((current_msdec['MSDECEndDateTime']
                            - current_msdec['MSDECStartDateTime'])
                            / pd.Timedelta(minutes=1))
    current_msdec = current_msdec.loc[current_msdec['LoS'] <= 720].copy()
    #Organise into age bands, put 90+ together
    banding = age_bands(current_msdec['PatientAge'])
    current_msdec['Age Bands'] = [band if int(re.findall(r'\d+', band)[0]) < 90
                                  else '90 and over' for band in banding[0]]
    
    #####LOS in MSDEC Chair (Based on mean and std of each age band)
    LoS = current_msdec.groupby(['Age Bands', 'Weekend'])['LoS'].agg(['mean', 'std'])
    print('LoS:')
    print(LoS.mean()/60)
    print('--------------')

    #####MSDEC Inter-arrivals using Population Projections
        ######Get the current numbers of daily arrivals
    current_msdec['Date'] = current_msdec['MSDECStartDateTime'].dt.date
    daily_arrivals = current_msdec[['Age Bands', 'Date', 'Weekend']].value_counts().reset_index()
    #Add in 0s for dates where age bands didn't see an attandence to avoid
    #daily figures being too high.
    dates = current_msdec['Date'].drop_duplicates().to_list()
    weekend = ['Weekend' if date.weekday() >= 5 else 'Weekday' for date in dates]
    age_bands = current_msdec['Age Bands'].drop_duplicates().to_list()
    crosstab = pd.DataFrame(itertools.product(dates, age_bands),
                            columns=['Date', 'Age Bands'])
    crosstab['Weekend'] = ['Weekend' if date.weekday() >= 5 else 'Weekday'
                           for date in crosstab['Date']]
    #Merge missing days onto data, fill with 0.  Group up to get average
    #arrivals by age band, multiply by 0.85 to remove the 15% UTC demand
    daily_arrivals = daily_arrivals.merge(crosstab,
                     on=['Age Bands', 'Date', 'Weekend'], how='outer').fillna(0)
    print(daily_arrivals.groupby(['Age Bands', 'Weekend'])['count'].mean().groupby('Weekend').sum())
    daily_arrivals = daily_arrivals.groupby(['Age Bands',
                                             'Weekend'])['count'].mean() * 0.85
    print(daily_arrivals.groupby('Weekend').sum())
    #Scale up or doqn  daily arrivals by change in opening hours
    scale = 1 + (MSDEC_close - MSDEC_open - 12)/12
    daily_arrivals = daily_arrivals * scale
    print(daily_arrivals.groupby('Weekend').sum())

        ######Read in the % change between years for each age group, add
        ######current arrival rates to the change dataframe
    change = pd.read_csv(
             'C:/Users/obriene/Projects/Discrete Event Simulation/MSDEC model/UHPT Population Change.csv'
             ).set_index('Age Bands')
    years = [str(i) for i in range(datetime.today().year,
                                   datetime.today().year + 11)]
    change = pd.DataFrame(daily_arrivals).join(change)
    #Loop through each age group and using current arrival numbers, use the
    #population projections to simulate number of arrivals in n years
    projections = []
    for row in change.values.tolist():
        start = row[0]
        props = row[1:]
        new_row = [start]
        for prop in props:
            start *= prop
            new_row.append(start)
        projections.append(new_row)
    #Create data frame and transform into inter arrival times, (max_LoS - 60 to
    #account for the fact that we stop arrivals 60 mins before close).
    daily_arrivals = pd.DataFrame(projections, columns=years, index=change.index)
    inter_arr = ((max_LoS-60) / daily_arrivals).round()

class spawn_patient:
    def __init__(self, p_id, age_band, age, arr_time, year, arr_day, arr_hour,
                 weekend, time_to_close):
        #Establish variables to store results
        self.id = p_id #patient id
        self.age_band = age_band #age band
        self.age = age #age
        self.arr_time = arr_time #arrival time
        self.arr_year = year #model year
        self.arr_day = arr_day #model day
        self.is_weekend = weekend
        self.arr_hour = arr_hour #model hour
        self.time_to_close = time_to_close
        self.dis_time = np.nan #discharge time
        
class msdec_model:
    def __init__(self, run_number, inputs):
        #Empty lists for results
        self.patient_results = []
        self.mau_occupancy_results = []
        #establish initial parameters
        self.env = simpy.Environment() #start environment
        self.inputs = inputs #set up input parameters
        self.patient_counter = 0 #set patient counter to 0
        self.run_number = run_number #record run number
        self.year = self.inputs.start_year #set start year
        #establish chair resource
        self.chair = simpy.Resource(self.env, capacity=self.inputs.no_chairs)
    
    ############################INCREASE YEAR###############################
    def increase_year(self):
        while True:
            yield self.env.timeout(365*24*60)
            self.year += 1

    #############################MODEL TIME#################################
    def model_time(self, time):
        #Work out what day and time it is in the model.
        day = math.floor(time / (24*60))
        day_of_week = day % 7
        weekend = 'Weekend' if day_of_week in [5, 6] else 'Weekday'
        #If day 0, hour is time / 60, otherwise it is the remainder time once
        #divided by number of days
        hour = math.floor((time % (day*(24*60)) if day != 0 else time) / 60)
        return day, day_of_week, weekend, hour
    
    ##########################MSDEC Open Close##############################
    def MSDEC_open(self, hour):
        #Function to work out if MSDEC is open or closed
        MSDEC_open = ((hour >= self.inputs.MSDEC_open)
                      and (hour < self.inputs.MSDEC_close))
        return MSDEC_open
    
    #############################ARRIVALS##################################
    def generate_age_band_arrivals(self, age_band):
        #Get average inter-arrival time for that age band
        av_inter_arrival = self.inputs.inter_arr.loc[(age_band, 'Weekday'),
                                                     str(self.year)]
        #Pick out limits for age band in order to randomly sample age
        ages = re.findall(r'\d+', age_band)
        ages = ([int(ages[0]), int(ages[1])] if len(ages) == 2 else [90, 105])
        #initial wait time of av_inter_arr when repeating this for all age bands
        # to prevent a rush at time 0
        yield self.env.timeout(av_inter_arrival)
        while True > 0:
            #Get model time variables
            time = self.env.now
            day, day_of_week, weekend, hour = self.model_time(time)
            #If MSDEC is not open, time out until it is, plus a random stagger
            #of a couple of hours to mimic actual distribution.
            if not self.MSDEC_open(hour):
                nxt_open_day = day if hour < self.inputs.MSDEC_close else day + 1
                nxt_open = (nxt_open_day)*(24*60) + (self.inputs.MSDEC_open*60)
                stagger = random.expovariate(1.0 / (60*5)) # randomly stagger
                #If stagger goes over close, re-sample until during opening hours
                while stagger > (self.inputs.MSDEC_close - self.inputs.MSDEC_open)*60:
                    stagger = random.expovariate(1.0 / (60*5)) # randomly stagger
                timeout = nxt_open - time + stagger

                yield self.env.timeout(nxt_open - time + stagger)

            #Ensure patient has at least an hour left before close, otherwise
            #wait until next day.
            time = self.env.now
            day, day_of_week, weekend, hour = self.model_time(time)
            time_to_close = (day*(24*60) + (self.inputs.MSDEC_close*60) - time)
            if time_to_close > 60:
                #up patient counter
                self.patient_counter += 1
                #randomly sample age
                age = random.randint(ages[0], ages[1])
                #Create patient and begin MSDEC journey
                p = spawn_patient(self.patient_counter, age_band, age, time,
                                self.year, day, hour, weekend, time_to_close)
                self.env.process(self.msdec_journey(p))
            #re-calculate inter arrival time and wait that time until next arrival
            av_inter_arrival = self.inputs.inter_arr.loc[(age_band, weekend),
                                                                str(self.year)]
            timeout = random.expovariate(1.0 / av_inter_arrival)
            yield self.env.timeout(timeout)

    ########################ED TO MAU PROCESS ##############################
    def msdec_journey(self, patient):
        #Patient comes into msdec and gets a chair
        with self.chair.request() as req:
            yield req
            #randomly sample the time spent in msdec based on age band LoS.
            mean, std = self.inputs.LoS.loc[(patient.age_band, patient.is_weekend)]
            sampled_LoS = min(np.random.normal(mean, std), patient.time_to_close)
            #If LoS falls outside of limits, repeat sample until it comforms.
            while ((sampled_LoS < self.inputs.min_LoS)
                   or (sampled_LoS > self.inputs.max_LoS)):
                    sampled_LoS = np.random.normal(mean, std)
            yield self.env.timeout(sampled_LoS)
        #Record discharge time and save patient results
        patient.dis_time = self.env.now
        self.store_patient_results(patient)

    ############################RECORD RESULTS##############################
    def store_patient_results(self, patient):
            self.patient_results.append([self.run_number, patient.id,
                                         patient.age_band, patient.age,
                                         patient.arr_time, patient.arr_year,
                                         patient.arr_day, patient.is_weekend,
                                         patient.arr_hour, patient.dis_time])
        
    def store_occupancy(self):
        while True:
            day, day_of_week, weekend, hour = self.model_time(self.env.now)
            self.mau_occupancy_results.append([self.run_number,
                                               self.chair._env.now, self.year,
                                               weekend,
                                               self.chair.count])
            yield self.env.timeout(self.inputs.occ_sample_time)

    ##################################RUN##################################
    def run(self):
        #Run processes, including a generator for each age band
        for age_band in set(self.inputs.LoS.index.get_level_values(0)):
            self.env.process(self.generate_age_band_arrivals(age_band))
        self.env.process(self.increase_year())
        self.env.process(self.store_occupancy())
        self.env.run(until=(self.inputs.run_time))
        #assign these back to default params
        default_params.pat_res += self.patient_results
        default_params.occ_res += self.mau_occupancy_results
        return self.patient_results, self.mau_occupancy_results
    
#################################FORMAT RESULTS#################################
def time_to_day_and_hour(col):
    #Functionn to get day and hour from model time
    day = col  // (24*60)
    hour = ((col / 60) % 24).apply(np.floor)
    return pd.DataFrame({'Day':day, 'Hour': hour})

def export_results(pat_results, occ_results):
    #Put full patient results into dataframe
    patient_df = (pd.DataFrame(pat_results,
                               columns=['run', 'pat ID', 'age band', 'age', 
                                        'arr time', 'arr year', 'arr day',
                                        'weekend', 'arr hour', 'dis time'])
                                       .sort_values(by=['run', 'pat ID']))
    patient_df[['arr day', 'arr hour']] = time_to_day_and_hour(
                                                        patient_df['arr time'])

    #Put occupaion output data into dataframe
    occ_df = pd.DataFrame(occ_results, columns=['run', 'time', 'year',
                                                'weekend', 'occ'])
    occ_df[['day', 'hour']] = time_to_day_and_hour(occ_df['time'])

    return patient_df, occ_df

#############################RUN FOR EACH ITERATION#############################
def run_the_model(inputs):
    #run the model for the number of iterations specified
    for run in range(inputs.iterations):
        print(f"Run {run+1} of {inputs.iterations}")
        model = msdec_model(run, inputs)
        model.run()
    patient_df, occ_df = export_results(inputs.pat_res, inputs.occ_res)
    return patient_df, occ_df

##################################SAVE RESULTS##################################
pat, occ = run_the_model(default_params)
os.chdir('C:/Users/obriene/Projects/Discrete Event Simulation/MSDEC model/Outputs')
pat.to_csv(f'Patients - {default_params.scenario_name}.csv')
occ.to_csv(f'Occupancy - {default_params.scenario_name}.csv')

################################################################################
                                ####PLOTS####
################################################################################
#####Daily Arrivals

#Daily arrivals over the next 10 years
daily_arr = (pat.groupby(['run', 'arr year', 'arr day', 'weekend'],
                         as_index=False)['pat ID'].count()
             .groupby(['arr year', 'weekend'], as_index=False)['pat ID'].mean()
             .pivot(index='arr year', columns='weekend', values='pat ID'))
daily_arr['Average'] = ((daily_arr['Weekday']*5) + (daily_arr['Weekend']*2)) / 7
daily_arr.plot(title='Daily MSDEC Arrivals for the Next 10 Years',
          figsize=(20, 15), grid=True,
          style={'Weekday':'b', 'Weekend':'g', 'Average':'--r'})
plt.savefig(f'Daily Arrivals - {default_params.scenario_name}.png', bbox_inches='tight')
plt.close()

#Daily arrivals over the next 10 years by age band (need to do final average
#manually to account for some days seeing no patients in that age band, otherwise
#outputs are too high)
av_day_arr = ((pat.groupby(['run', 'age band', 'arr year', 'arr day'],
                           as_index=False)['pat ID'].count()
                  .groupby(['arr year', 'age band'],
                           as_index=False)['pat ID'].sum()))
av_day_arr['pat ID'] = (av_day_arr['pat ID'] / 365) / default_params.iterations

av_day_arr.pivot(columns='age band', index='arr year', values='pat ID').plot(
    figsize=(20, 15), title=av_day_arr['age band'].drop_duplicates().to_list(),
    subplots=True, layout=(4, 4), xlabel='Year',  ylabel='Daily Arrivals',
    ylim=(0,np.ceil(av_day_arr['pat ID'].max())), legend=False, grid=True,
    colormap='winter')
plt.savefig(f'Daily Arrivals by Age Band - {default_params.scenario_name}.png', bbox_inches='tight')
plt.close()

######Occupancy
#remove first day warm up time
wk_occ = occ.loc[(occ['day'] > 0) & (occ['occ'] > 0)].copy()
wd_occ = wk_occ.loc[wk_occ['weekend'] == 'Weekday'].copy()
we_occ = wk_occ.loc[wk_occ['weekend'] == 'Weekend'].copy()

#Set formats
whis = (5, 95)
boxprops = dict(linestyle='-', linewidth=3, color='black')
whiskerprops = dict(linestyle='-', linewidth=3, color='black')
capprops = dict(linestyle='-', linewidth=3, color='black')
medianprops = dict(linestyle='-', linewidth=2.5, color='firebrick')
meanlineprops = dict(linestyle='--', linewidth=2.5, color='firebrick')
flierprops = dict(marker='o', markerfacecolor='grey', markersize=5, markeredgecolor='none')

for name, df in [('Overall', wk_occ), ('Weekdays', wd_occ), ('Weekends', we_occ)]:
    #Create box plot for each year
    columns = df['year'].drop_duplicates().to_list()
    fig, ax = plt.subplots(figsize=(20,10))
    for position, column in enumerate(columns):
        bp = ax.boxplot(df.loc[df['year'] == column, 'occ'], positions=[position],
                        sym='.', widths=0.9, whis=whis, showmeans=True,  meanline=True,
                        boxprops=boxprops, medianprops=medianprops,
                        whiskerprops=whiskerprops,
                        flierprops=flierprops, capprops=capprops,
                        meanprops=meanlineprops)
    #Create and save figure
    ax.set_yticks(range(df['occ'].max()+1))
    ax.set_xticklabels(columns, fontdict={'fontsize':20})
    ax.set_xlim(xmin=-0.5)
    ax.set_title(f'{name} MSDEC Chair Occupancy Box Plots by Year ({whis[0]}% - {whis[1]}%)', fontdict={'fontsize':20})
    plt.legend([bp['medians'][0], bp['means'][0]], ['median', 'mean'])
    ax.grid()
    plt.savefig(f'{name} MSDEC Chair Occupancy Box Plots by Year - {default_params.scenario_name}.png', bbox_inches='tight')
    plt.close()


#####Occupancy Hour of Day plot
#Metrics by hour of day
# 25th and 75th Percentiles functions 
def q25(x):
    return x.quantile(0.25)
def q75(x):
    return x.quantile(0.75)
def q80(x):
    return x.quantile(0.80)
def q85(x):
    return x.quantile(0.85)
def q90(x):
    return x.quantile(0.90)
def q95(x):
    return x.quantile(0.95)

occ_metrics = (occ.groupby(['weekend', 'hour'], observed=False,  as_index=False)['occ']
               .agg(['min', q25, 'mean', q75, q80, q85, q90, q95, 'max']))
hours = occ_metrics['hour'].drop_duplicates()
#plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10), sharey=True)
fig.suptitle('Occupancy by Hour of Day', fontsize=24)
wd_metrics = occ_metrics.loc[occ_metrics['weekend'] == 'Weekday'].copy()
ax1.plot(hours, wd_metrics['mean'].fillna(0), '-r', label='Mean')
ax1.fill_between(hours, wd_metrics['min'].fillna(0), wd_metrics['max'].fillna(0), color='grey', alpha=0.2, label='Min-Max')
ax1.fill_between(hours, wd_metrics['q25'].fillna(0), wd_metrics['q75'].fillna(0), color='black', alpha=0.2, label='LQ-UQ')
ax1.set_title('Weekday', fontsize=18)
ax1.set_xlabel('Hour of Day', fontsize=18)
ax1.set_ylabel('No. Chairs Occupied', fontsize=18)
ax1.tick_params(axis='both',  which='major', labelsize=18)
ax1.legend(fontsize=18)
we_metrics = occ_metrics.loc[occ_metrics['weekend'] == 'Weekend'].copy().merge(hours, how='outer').fillna(0)
ax2.plot(hours, we_metrics['mean'], '-r')
ax2.fill_between(hours, we_metrics['min'], we_metrics['max'], color='grey', alpha=0.2)
ax2.fill_between(hours, we_metrics['q25'], we_metrics['q75'], color='black', alpha=0.2)
ax2.set_title('Weekend', fontsize=18)
ax2.set_xlabel('Hour of Day', fontsize=18)
ax2.tick_params(axis='both',  which='major', labelsize=18)
fig.tight_layout()
plt.savefig(f'Hourly Occupancy Weekday and Weekend - {default_params.scenario_name}.png',
            bbox_inches='tight', dpi=1200)
plt.close()

print('-----------------------')
print('Daily Arrivals Quartiles:')
#Print the quartiles for the number of arrivals in the model
print((pat.loc[pat['weekend'] == 'Weekday']
       .groupby(['run', 'arr year', 'arr day'],as_index=False)['pat ID'].count()
       .groupby('arr year')['pat ID']
       .agg(['min', q25, 'mean', q75, q80, q85, q90, q95, 'max'])).round(2))
print('-----------------------')
print('Daily Occupancy Quartiles:')
#Get only occupancy during opening hours on a week day, as this is higher than
#weekends.  Print the quartiles
open_occ = occ.loc[(occ['hour'] > default_params.MSDEC_open)
                   & (occ['hour'] < default_params.MSDEC_close)
                   & (occ['weekend'] == 'Weekday')].copy()
occ_sum = (open_occ.groupby('year')['occ']
           .agg(['min', q25, 'mean', q75, q80, q85, q90, q95, 'max']))
print(occ_sum)

print('-----------------------')
print('Arrivals Based on Occupancy Quartiles:')
#Find the days where occupancy reaches those quartiles.
now = occ_sum.iloc[0]
fut = occ_sum.iloc[-1]
def arrivals_for_quartiles(row, label):
    arrs = []
    for quart in row.round():
        days = occ.loc[(occ['occ'] == quart) & (occ['year'] == int(label)),
                       ['run', 'day']].drop_duplicates()
        arrivals = (days.merge(pat, left_on=['run', 'day'], 
                               right_on=['run', 'arr day']
                               ).groupby(['run', 'day'])['pat ID']
                               .count().mean().round(2))
        arrs.append(arrivals)
    out = pd.DataFrame([arrs], columns=row.index, index=[label])
    return out

print(arrivals_for_quartiles(now, '2025'))
print(arrivals_for_quartiles(fut, '2034'))
print('-----------------------')