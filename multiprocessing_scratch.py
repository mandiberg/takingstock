from multiprocessing import Lock, Process, Queue, current_process
import time
import queue # imported for using queue.Empty exception
import csv


'''

select 1k images from database with no is_face variable

SELECT column_names
FROM table_name
WHERE column_name IS NULL; 

send each row into the 
        tasks_to_accomplish.put("Task no " + str(i))

i think i want the whole row:
 i need the UID to add the data back, and the filepath to read it
 so easiest just to send the whole thing in?

do_job will
mediapipe post est
write is_face and landmarks
write calc data

'''

csv_file = "/Users/michaelmandiberg/Documents/projects-active/facemap_production/_SELECT_FROM_faceimages_query_mouthopen.csv"

def read_csv(csv_file):
    with open(csv_file, encoding="utf-8", newline="") as in_file:
        reader = csv.reader(in_file, delimiter="|")
        next(reader)  # Header row

        for row in reader:
            yield row


def do_job(tasks_to_accomplish, tasks_that_are_done):
    while True:
        try:
            '''
                try to get task from the queue. get_nowait() function will 
                raise queue.Empty exception if the queue is empty. 
                queue(False) function would do the same task also.
            '''
            task = tasks_to_accomplish.get_nowait()
        except queue.Empty:

            break
        else:
            '''
                if no exception has been raised, add the task completion 
                message to task_that_are_done queue
            '''
            print(task)
            print("this is where you do stuff?")
            tasks_that_are_done.put(task + ' is done by ' + current_process().name)
            time.sleep(.5)
    return True


def main():
    number_of_task = 10
    number_of_processes = 8
    tasks_to_accomplish = Queue()
    tasks_that_are_done = Queue()
    processes = []

    for row in read_csv(csv_file):
    	print(row)

    for i in range(number_of_task):
        tasks_to_accomplish.put("Task no " + str(i))

    # creating processes
    for w in range(number_of_processes):
        p = Process(target=do_job, args=(tasks_to_accomplish, tasks_that_are_done))
        processes.append(p)
        p.start()

    # completing process
    for p in processes:
        p.join()

    # print the output
    while not tasks_that_are_done.empty():
        print(tasks_that_are_done.get())

    return True


if __name__ == '__main__':
    main()



