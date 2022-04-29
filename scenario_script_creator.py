import scenario_dict
for scenario in scenario_dict.scenario_dict.keys():
    output_filename = f'batch_scripts/{scenario}.sh'
    filename = 'script.sh'
    with open(filename) as f:
        newText=f.read().replace('scenario', scenario)

    with open(output_filename, "w") as f:
        f.write(newText)