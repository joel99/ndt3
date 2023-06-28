% Batch convert Pitt data
% Usage:
%   Set `root_dir` below
%   Open matlab shell and add rtma/climber to path (or e.g. init in a dir with appropriate startup.m)
%   cd to this working dir (so that prep_all is available)
%   Call prep_all

function prep_all(subject)

    % Prepares all experiment data for a given subject.
    % Example usage:
    %     prep_all('CRS08_14')
    %
    % Parameters:
    %     subject (string): The subject identifier to filter experiment data on.
    %                       For example, 'crs08' for subject 'crs08'.
    %
    % Returns:
    %     None

    root_dir = "/home/joelye/projects/context_general_bci/data/pitt_misc/";
    experiments = dir(root_dir);
    subject_id = strcat(subject, '_');
    for experiment = experiments'
        if startsWith(experiment.name, subject_id)
            experiment = experiment.name
            disp(experiment)
            prefix = "QL.";
            experiment_dir = fullfile(root_dir, experiment);
            ql_files = dir(fullfile(experiment_dir,strcat(prefix, '*.bin')));
            target_dir = fullfile(root_dir, "mat", experiment)
            names = arrayfun(@(x) x.name, ql_files, 'UniformOutput', false);
            prep_multiple(root_dir, experiment, names)
        end
    end
