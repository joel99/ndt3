% Batch convert Pitt data
% Usage:
%   Set `root_dir` below
%   Open matlab shell and add rtma/climber to path (or e.g. init in a dir with appropriate startup.m)
%   cd to somewhere where prep_all is available
%   Call prep_all

function prep_all

% root_dir = "/home/joelye/user_data/stim_team/s1_m1";
% root_dir = "/home/joelye/projects/icms_modeling/data/detection";
% root_dir = "/home/joelye/projects/context_general_bci/data/sample_crs02b_data/"
% root_dir = "/home/joelye/projects/context_general_bci/data/pitt_co/"
root_dir = "/home/joelye/projects/context_general_bci/data/pitt_misc/"
experiments = dir(root_dir);
for experiment = experiments'
    if startsWith(experiment.name, "crs08")
        experiment = experiment.name
        disp(experiment)
        % experiment = "CRS02bHome.data.00329";
        prefix = "QL.";
        experiment_dir = fullfile(root_dir, experiment);
        ql_files = dir(fullfile(experiment_dir,strcat(prefix, '*.bin')));
        target_dir = fullfile(root_dir, "mat", experiment)
        mkdir(target_dir)
        % disp(ql_files)
        names = arrayfun(@(x) x.name, ql_files, 'UniformOutput', false);
        prep_multiple(root_dir, experiment, names)
        % for k = 1:length(ql_files)
            % if startsWith(ql_files(k).name, "QL.Task")
                % prep_and_cast(root_dir, experiment, ql_files(k).name)
            % end
        % end
        % break
    end
end