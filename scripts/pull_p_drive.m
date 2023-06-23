% queries = {'2D Cursor Center Out', 'free play', 'overt movement'};
% queries = {'Free Play'};
% queries = {'Cursor Center-Out', 'Cursor Center Out', 'Helicopter Rescue'};
queries = {'Center-Out'}; % Note this query isn't really on the head - there are many center-out related tasks that don't use cursor, and many cursor CO that aren't phrased exactly like this. Nonetheless, this is our first pass....
% queries = {'Free Play'};

blacklist_comments = ["abort", "discard", "ignore", "trash", "wrong"]; % in lieu of abort, discard, ignore, trash
for i = 1:length(queries)
    disp(queries{i});
    root_path_out = 'C:/Users/joy47/Desktop/pitt_varied';
    if contains(queries{i}, 'Center-Out')
        root_path_out = 'C:/Users/joy47/Desktop/pitt_co';
    end
    res = searchAllLogs(queries{i}, 'subject', 'CRS07');
%     res = searchLogsInRange(queries{i}, 'subject', 'CRS02b');
%     res = searchAllLogs(queries{i}, 'subject', 'CRS02b');

    if isfield(res, 'sets')
%     [days, sets] = searchAllLogs(queries{i});
        for j = 1:length(res.sets)
            set = res.sets(j);

            if ~contains(set.paradigm, queries{i}) || ~contains(set.paradigm, '2D')
                continue;
            end

            paradigm = lower(set.paradigm);
            paradigm = strrep(paradigm, '-', ' ');
            paradigm = strrep(paradigm, ' ', '_');

            is_valid = true;
%             checks
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

            if strcmp(queries{i}, 'free play') && ~strcmp(paradigm, 'free_play')
                is_valid = false;
            end

            type_tag = paradigm;

            try
                comments = set.comments;
                if strcmp(paradigm, 'free play') || strcmp(paradigm, 'free-play')
                    type_tag = [type_tag, '_free_play'];
                elseif contains(comments, 'ortho')
                    type_tag = [type_tag, '_ortho'];
                elseif contains(comments, 'fbc') || contains(comments, 'full')
                    type_tag = [type_tag, '_fbc'];
                elseif contains(comments, 'obs')
                    type_tag = [type_tag, '_obs'];
                elseif ~contains(paradigm, 'center') % don't get toooo heterogeneous, restrict to center out
                    is_valid = false;
                end
            catch
                continue % some failures on set.paradigm lower for empty str, IDK syntax to address but that'd effectively be invalid anyway
            end

            if ~is_valid
                continue
            end
            try
                [data] = set.loadSetData();
            catch
                continue
            end
            set_name = convertCharsToStrings(set.sessionObj.dayObj.subjectID) + set.sessionObj.dayObj.location + ...
                "_session_" + num2str(set.sessionObj.sessionNum) + ...
                "_set_" + num2str(set.setNum) + ...
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
                    if size(thin_data.pos, 1) ~= size(thin_data.SpikeCount, 1)
                        disp("mismatched shape, drop " + set_name);
                        clearvars thin_data.pos; % abandon attempt
                        clearvars thin_data.target; % abandon attempt
                    end
                end
                save(out_path, 'thin_data');
            catch
                continue
            end
        end
    end
end