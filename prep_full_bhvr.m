% Batch convert manually transported Pitt data
% Usage:
%   Set `root_dir` below
%   Open matlab shell and add rtma/climber to path (or e.g. init in a dir with appropriate startup.m)
%   cd to somewhere where prep_all is available
%   Call prep_all(prefix)

function prep_full_bhvr(subject)

root_dir = char(pwd);
root_dir = fullfile(root_dir, "data/pitt_misc/");
experiments = dir(root_dir)
for experiment = experiments'
    disp(experiment.name)
    % add breakpoint

    if startsWith(experiment.name, subject)
        tag = experiment.name
        prefix = "QL.";
        experiment_dir = fullfile(root_dir, tag);
        ql_files = dir(fullfile(experiment_dir,strcat(prefix, '*.bin')));
        target_dir = fullfile(root_dir, "mat", tag);
        % mkdir(target_dir)
        % disp(ql_files)
        names = arrayfun(@(x) x.name, ql_files, 'UniformOutput', false);
        prep_full_bhvr_single(root_dir, tag, names)
    end
end