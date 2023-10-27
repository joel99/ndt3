queries = {'2D Cursor Center Out', 'free play', 'overt movement'};

root_path_out = '~/data/pitt_varied/'

for query in queries'
    [days, sets] = searchAllLogs(query);
    for set in sets'
        [data] = set.loadSetData();
        set_name = set.sessionObj.sessionNum + '_' + set.setNum;
        out_path = [root_path_out set_name '.mat'];
        save(root_path_out, 'data');
    end