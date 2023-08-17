% For converting Pitt QL format to matlab objects (to then be ingested by my python libs..)
% Requires Pitt RTMA lib in matlab path

function prep_multiple_for_analysis(root_dir, experiment, files)
% Load quicklogger trial, cast StimData to a struct, and save as processed matlab

src_files = cellfun(@(file) fullfile(root_dir, experiment, file), files, 'UniformOutput', false);
src_files_char = cellfun(@char, src_files, 'UniformOutput', false);
% disp(src_files)
[data, iData] = prepData('files', src_files_char);
% keyboard
%  Pitt patch for manual conversion to mimic pull_p_drive
% Split the string based on underscore delimiter
parts = split(experiment, '_');
% Extract the session and set integers, and type as string
subject_name = parts(1);
session = str2double(parts(2));
set = str2double(parts(3));
% type_tag = parts(4);
type_tag = 'fbc'; % typically we are analyzing fbc, and we don't have this info in the filename

root_path_out = fullfile(root_dir, 'mat');

set_name = subject_name + ... % Implied location
    "_session_" + num2str(session) + ...
    "_set_" + num2str(set) + ...
    "_type_" + type_tag;
out_path = fullfile(root_path_out, set_name + ".mat");
try
    thin_data = [];
    thin_data.SpikeCount = cast(data.SpikeCount, 'uint8');
    thin_data.SpikeCount = thin_data.SpikeCount(:,1:5:end); % only get unsorted positions (we don't sort)
    thin_data.trial_num = cast(data.trial_num, 'uint8');
    if endsWith(type_tag, 'fbc') || endsWith(type_tag, 'ortho') || endsWith(type_tag, 'obs')
%                 if strcmp(type_tag, 'fbc') || strcmp(type_tag, 'ortho') || strcmp(type_tag, 'obs')
        thin_data.pos = cast(data.Kinematics.ActualPos(:,1:3), 'single');
        thin_data.target = cast(data.TaskStateMasks.target(1:3, :), 'single');
        thin_data.task_states = cast(data.TaskStateMasks.state_num, 'uint8');
        thin_data.state_strs = data.TaskStateMasks.states;
        % keyboard
        thin_data.passed = data.XM.passed;
        thin_data.failed = data.XM.failed;
        if size(thin_data.pos, 1) ~= size(thin_data.SpikeCount, 1)
            disp("mismatched shape, drop " + set_name);
            clearvars thin_data.pos; % abandon attempt
            clearvars thin_data.target; % abandon attempt
        end
    end
    save(out_path, 'thin_data');

end
