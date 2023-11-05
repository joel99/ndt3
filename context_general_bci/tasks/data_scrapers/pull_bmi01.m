% Indiscriminate Pegboard pull for BMI01 who has special paths
% V 0.1
function pull_bmi01(tag)
% Tag as BMI01Lab or BMI01Home
root_path_out = 'D:/Data/pitt_bmi01';
if ~exist(root_path_out, 'dir')
    mkdir(root_path_out);
end

% root_paths_in = {
% 'P:/data_raw/human/rp3_bmi/BMI01/Pegboard/BM01Lab/';
% 'P:/data_raw/human/rp3_bmi/BMI01/Pegboard/BM01Home/'
% };
% root_path_in = 'P:/data_raw/human/rp3_bmi/BMI01/Pegboard/BMI01Lab/'; % See loadData wildcard building
root_path_in = fullfile('P:/data_raw/human/rp3_bmi/BMI01/Pegboard/', tag); % See loadData wildcard building
% Assuming the directory listing should return subdirectories,
% we use the 'dir' function with a folder wildcard
session_paths = dir(fullfile(root_path_in, 'BMI*'));
for idx = 1:length(session_paths)
    if session_paths(idx).isdir
        session_name = session_paths(idx).name;
        session_path = fullfile(session_paths(idx).folder, session_name);

        % Process by set, not full session, as we don't want blocks of
        % inactive times
        % between sets

        % Get list of all files with "SetXXXX" in their name
        fileList = dir(fullfile(session_path, '*Set*bin'));

        % Extract set numbers and find unique ones
        setNumbers = cellfun(@(x) regexp(x, 'Set(\d+)', 'tokens'), {fileList.name}, 'UniformOutput', false);
        setNumbers = [setNumbers{:}];
        if isempty(setNumbers)
            continue;
        end
        flattenedStrings = cellfun(@(x) x{1}, setNumbers, 'UniformOutput', false);

        % Convert the flattened strings to numbers
        setNumbers = cellfun(@str2double, flattenedStrings);
%         setNumbers = cellfun(@(x) str2double(x{1}{1}), setNumbers);

        % Find unique sets
        uniqueSets = unique(setNumbers);

        for i = 1:length(uniqueSets)
            session_set = uniqueSets(i);
            % Your code here for processing each set
            try
                paddedSetNumber = sprintf('%04s', num2str(session_set)');
                full_path = [session_path, filesep, 'QL.Task_State*Set', paddedSetNumber, '*.bin'];
                data = prepData('files', full_path); % Ensure the prepData function is properly defined in MATLAB
            catch e
                fprintf(1,'Fail prep data:\n%s',e.identifier);
                fprintf(1,'Fail prep data:\n%s',e.message);
                continue; % Skip to the next iteration
            end
            session_no = regexp(session_path, 'BMI01.*\.data\.(\d+)$', 'tokens', 'once');
            session_no = cell2mat(session_no);
            session_no = str2num(session_no);
            out_filename = [tag, '_session_', num2str(session_no), '_set_', num2str(session_set), '.mat'];
            out_path = fullfile(root_path_out, out_filename);
            try
                thin_data = struct(); % Initialize an empty struct
                % No stim for BMI01
    %             thin_data.SpikeCount = uint8(data.SpikeCount);
    %             thin_data.SpikeCount = thin_data.SpikeCount(:, 1:5:end); % only get unsorted positions (we don't sort)
                % Some of BMI01's data is sorted! Sum up to unsort
                n = size(data.SpikeCount, 2);
                groupSize = 5;
                numGroups = n / groupSize;
                per_channel_data = reshape(data.SpikeCount, [], groupSize, numGroups);
                per_channel_data = squeeze(sum(per_channel_data, 2));
                thin_data.SpikeCount = uint8(per_channel_data);

                thin_data.trial_num = uint8(data.trial_num);
                thin_data.passed = data.XM.passed;
                % NOTE: You might want to save the 'thin_data' to 'out_path' using the 'save' function
                % save(out_path, 'thin_data');

                if isfield(data.Kinematics, 'ActualPos') || isfield(data.Kinematics, 'ActualForce')
        %                 if endsWith(type_tag, 'fbc') || endsWith(type_tag, 'ortho') || endsWith(type_tag, 'obs')
                    if isfield(data.Kinematics, 'ActualPos')
                        thin_data.pos = cast(data.Kinematics.ActualPos(:,1:14), 'single'); % BMI01 did up to 10D, and none in force
                        if size(thin_data.pos, 1) ~= size(thin_data.SpikeCount, 1)
                            disp("mismatched shape, drop " + set_name);
                            clearvars thin_data.pos; % abandon attempt
                        end
                    end

                    if isfield(data.Kinematics, 'ActualForce')
                        thin_data.force = cast(data.Kinematics.ActualForce(:,1:1), 'single'); % Though this is 5D, I'm told by GB that essentially only dim 1 (index finger point force sensor in Mujoco) is rendered
                    end
                    % 1:3 - only take right hand's worth of control domains
                    % (translation, rotation, grasp)
                    thin_data.brain_control = cast(data.TaskStateMasks.brain_control_weight(1:3, :)', 'single');
                    thin_data.brain_control(isnan(thin_data.brain_control)) = 0;
                    thin_data.active_assist = cast(data.TaskStateMasks.active_assist_weight(1:3, :)', 'single');
                    thin_data.active_assist(isnan(thin_data.active_assist)) = 0;
                    thin_data.passive_assist = cast(data.TaskStateMasks.passive_assist_weight(1:3, :)', 'single');
                    thin_data.passive_assist(isnan(thin_data.passive_assist)) = 0;
                    if isfield(data.TaskStateMasks, 'active_override')
                        if sum(data.TaskStateMasks.active_override(1:14, :), "all", "omitnan") > 0
                            thin_data.override = cast(data.TaskStateMasks.active_override(1:14, :)', 'single');
                            thin_data.override(isnan(thin_data.override)) = 0;
                        end
                    end
                end
                save(out_path, 'thin_data');
            catch e
                fprintf(1,'Fail save:\n%s',e.identifier);
                fprintf(1,'Fail prep data:\n%s',e.message);
                % Handle error or just continue to next iteration
                continue;
            end
        end

    end
end

end