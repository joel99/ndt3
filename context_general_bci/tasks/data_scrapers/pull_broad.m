% Near complete pull for NDT3, scraping all historical motor data.
% V 0.1
function pull_broad(subject)
% dbstop if warning
% dbclear all
queries = { ...
    'Cursor', 'MotorExperiments', 'Helicopter Rescue', 'Free Play', 'FreePlay', ...
    'Overt', 'Covert', 'G&C', 'Carry', ... % Carry as in Grasp & Carry
    'DOF hand control', 'Pursuit', 'Object', ... % Object as in object interaction and object transfer
    '1DOF Position Matching', 'Center-Out', 'ARAT', ...
    'Distraction', ...
    'Grasp'
    % Ignored: Left-only 2D, Right only 2D, bilateral lowDl
    };
% queries = {'4D G&C'};
% queries = {'Helicopter Rescue'};
blacklist_comments = ["abort", "discard", "ignore", "trash", "wrong", "very poor", "no control", "terrible", "debug", "error"]; % in lieu of abort, discard, ignore, trash
for i = 1:length(queries)
    disp(queries{i});
    root_path_out = 'D:/Data/pitt_broad';
    res = searchAllLogs(queries{i}, 'subject', subject);
    if isfield(res, 'sets')
        for j = 1:length(res.sets)
            set = res.sets(j);

            paradigm = lower(set.paradigm);

            if size(paradigm, 1) == 0 || ...
                    ~contains(set.paradigm, queries{i}) || ...
                    contains(paradigm, 'wmp') || ...
                    contains(paradigm, 'stim') || ...
                    contains(paradigm, 'icms') || ...
                    contains(paradigm, 'fofix') || ...
                    contains(paradigm, 'fragile') || ...
                    contains(paradigm, 'force adjust') % Too many visual conditions.
                continue;
            end

            paradigm = strrep(paradigm, '-', ' ');
            paradigm = strrep(paradigm, '+stitching', '');
            paradigm = strrep(paradigm, '+stitch', '');
            paradigm = strrep(paradigm, '+ stitching', '');
            paradigm = strrep(paradigm, '+ stitch', '');
            paradigm = strrep(paradigm, ' ', '_');
            paradigm = strip(paradigm);

            % Run blacklist checks
            is_valid = true;
            set.comments = set.comments';
            set.comments = set.comments(:)';
            set.comments = lower(set.comments);

            for b = 1:length(blacklist_comments)
                if contains(set.comments, blacklist_comments(b))
                    is_valid = false;
                    break;
                end
            end

            if ~is_valid
                continue;
            end

            try
                [data] = set.loadSetData();
            catch
                continue
            end
                set_name = convertCharsToStrings(set.sessionObj.dayObj.subjectID) + set.sessionObj.dayObj.location + ...
                "_session_" + num2str(set.sessionObj.sessionNum) + ...
                "_set_" + num2str(set.setNum);
            out_path = fullfile(root_path_out, set_name + ".mat");
            try
                thin_data = [];
                % More blacklist criteria
                if any(data.stim_idx)
                    continue
                end
                thin_data.SpikeCount = cast(data.SpikeCount, 'uint8');
                thin_data.SpikeCount = thin_data.SpikeCount(:,1:5:end); % only get unsorted positions (we don't sort)
                thin_data.trial_num = cast(data.trial_num, 'uint8');
                thin_data.passed = data.XM.passed;

                % We assume brain control weight is 1 - active assist

                % TODO cast NaN
                % TODO check what to scrape
                % Want to add constraints and auto - passive/active assist
                % weightj
                % need some heuristic to identify something about relevant
                % dims
                % should maybe store paradigm somewhere
                if isprop(set, 'endEffector')
                    thin_data.effector = lower(set.endEffector);
                end

                if isfield(data.Kinematics, 'ActualPos') || isfield(data.Kinematics, 'ActualForce')
%                 if endsWith(type_tag, 'fbc') || endsWith(type_tag, 'ortho') || endsWith(type_tag, 'obs')
                    if isfield(data.Kinematics, 'ActualPos')
                        thin_data.pos = cast(data.Kinematics.ActualPos(:,1:14), 'single'); % Can go up to 14 for finger
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
                    if sum(thin_data.brain_control, 'all') > 0
                        disp('hi')
                    end
                    if isfield(data.TaskStateMasks, 'active_override')
                        if sum(data.TaskStateMasks.active_override(1:14, :), "all", "omitnan") > 0
                            thin_data.override = cast(data.TaskStateMasks.active_override(1:14, :)', 'single');
                            thin_data.override(isnan(thin_data.override)) = 0;
                        end
                    end
                end
                save(out_path, 'thin_data');
            catch
                continue
            end
        end
    end
end

end