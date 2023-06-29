% Batch convert Pitt data
% Usage:
%   Set `root_dir` below
%   Open matlab shell and add rtma/climber to path (or e.g. init in a dir with appropriate startup.m)
%   cd to somewhere where prep_all is available
%   Call prep_all

function prep_all(subject)

root_dir = char(pwd);
root_dir = fullfile(root_dir, "/data/pitt_misc/");
experiments = dir(root_dir);
for experiment = experiments'
    if startsWith(experiment.name, subject)
        experiment = experiment.name
        disp(experiment)
        prefix = "QL.";
        experiment_dir = fullfile(root_dir, experiment);
        ql_files = dir(fullfile(experiment_dir,strcat(prefix, '*.bin')));
        target_dir = fullfile(root_dir, "mat", experiment)
        % mkdir(target_dir)
        % disp(ql_files)
        names = arrayfun(@(x) x.name, ql_files, 'UniformOutput', false);
        prep_multiple_for_analysis(root_dir, experiment, names)
    end
end