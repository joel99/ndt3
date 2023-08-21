% For converting Pitt QL format to matlab objects (to then be ingested by my python libs..)
% Requires Pitt RTMA lib in matlab path

function prep_and_cast(root_dir, experiment, file)
% Load quicklogger trial, cast StimData to a struct, and save as processed matlab

src_file = fullfile(root_dir, experiment, file)
[data, iData] = prepData('files', [src_file])
% [iData] = prepData('files', [src_file], 'format_data', false) # doesn't work for some reason...
% data.StimData = struct(data.StimData)
% iData.StimData = struct(iData.StimData)
target_dir = fullfile(root_dir, "mat", experiment)
[dummy, file_stem, ext] = fileparts(src_file)
target_file = fullfile(target_dir, strcat(file_stem, ".mat"))
% save(target_file, 'iData')
% save(target_file, 'iData', 'data')

%  Pitt hotpatch for manual conversion to mimic pull_p_drive
% Split the string based on underscore delimiter
parts = split(experiment, '_');
disp(parts);

% Extract the session and set integers, and type as string
subject_name = parts(1);
session = str2double(parts(2));
set = str2double(parts(3));
type_tag = parts(4);

disp(session);
disp(set);
disp(type_tag);

root_path_out = fullfile(root_dir, 'mat');

set_name = subject_name + ...
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
        thin_data.target = cast(data.TaskStateMasks.target(1:3), 'single');
        if size(thin_data.pos, 1) ~= size(thin_data.SpikeCount, 1)
            disp("mismatched shape, drop " + set_name);
            clearvars thin_data.pos; % abandon attempt
            clearvars thin_data.target; % abandon attempt
        end
    end
    save(out_path, 'thin_data');

end

