dir="test_results_jchVM_kLOTS"
mkdir -p ${dir}

END=10
timesteps=(200 300 400) # (200 300 400)
kappas=(0.001 0.01 0.1 1 5 10 20 30 40 50 60 70 80 90 100 110 120 130 150 200 250) # (0.01 1 10 100)

# loop length = # of positional arguments in timesteps
for (( i=0; i<${#timesteps[@]}; i++ )); do
    time=${timesteps[$i]}
    kappa=${kappas[$i]}
    for kappa in ${kappas[*]}; do
        folder="${dir}/time_${time}_kappa_${kappa}" 
        mkdir -p ${folder}
        #out1="results/time_${time}_kappa_${kappa}"
        #mkdir -p ${out1}
        for k in $(seq 1 $END); do # repeat each param combo 10 times
            #out2="${out1}/${k}"
            name="${folder}/${k}"
            echo $name
            python create_config.py --timesteps $time --kappa $kappa --seed $k --output $name > "${name}.config"
            sbatch run_p.sh "${name}.config"
        done
    done
done
