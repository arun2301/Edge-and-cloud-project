import pandas as pd
df = pd.read_csv('../results/energy_log_gpu.txt', sep='\s+', comment='#',
                  header=None,
                  names=['date','time','pwr','gtemp','mtemp','sm','mem','enc','dec','jpg','ofa','rx','tx'])
print(df[['time','pwr','sm','mem']].describe())
# pwr = power in Watts, sm = GPU core utilisation %