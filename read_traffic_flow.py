import numpy as np
import pandas as pd
import os
import datetime


def read_traffic_flow(files_dir, with_weekends=True):
    traffic_flow = []
    traffic_flow_train = []
    traffic_flow_test = []
    xlsxes = [os.path.join(files_dir, i) for i in os.listdir(files_dir)
              if i[-5:] == ".xlsx"]
    for xlsx in xlsxes:
        df = pd.read_excel(xlsx)
        traffic_flow.append(np.array([df["5 Minutes"],
                                      df["Flow (Veh/5 Minutes)"]]).T)
    traffic_flow = np.vstack(traffic_flow)
    dates = []
    for date in traffic_flow[:, 0]:
        date = date.split()[0]
        if date not in dates:
            dates.append(date)
    if not with_weekends:
        first_date = datetime.datetime.strptime('8/28/2017', '%m/%d/%Y')  # Mon
        for i_data in range(traffic_flow.shape[0]-1, -1, -1):
            curr_day = traffic_flow[i_data][0].split()[0]
            if not with_weekends:
                curr_date = datetime.datetime.strptime(
                    traffic_flow[i_data][0].split()[0], '%m/%d/%Y')
                if (curr_date - first_date).days % 7 > 4:
                    traffic_flow = np.delete(traffic_flow, i_data, 0)
                    # print("curr_day removed:", curr_day)
                    if curr_day in dates:
                        dates.remove(curr_day)
    # print("dates:", dates)
    for data in traffic_flow:
        curr_day = data[0].split()[0]
        # if curr_day == dates[-3]:
        #     traffic_flow_validate.append(data[1])
        if curr_day == dates[-2]:
            traffic_flow_test.append(data[1])
        elif curr_day == dates[-1]:
            continue
        else:
            traffic_flow_train.append(data[1])

    traffic_flow_train = np.array(traffic_flow_train)
    traffic_flow_test = np.array(traffic_flow_test)
    print("read_traffic_flow " + files_dir + "finished!")
    return traffic_flow_train, traffic_flow_test


def main():
    files_dir = "./traffic_flow_data/traffic data2017.9.28/Traffic flow/"
    data_train, data_test = read_traffic_flow(files_dir)
    print("traffic_flow_train.shape = " + str(data_train.shape))
    print("traffic_flow_test.shape = " + str(data_test.shape))


if __name__ == '__main__':
    main()
