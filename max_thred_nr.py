import threading

def test_max_threads():
    threads = []
    i = 0
    try:
        while True:
            t = threading.Thread(target=lambda: time.sleep(1))
            t.start()
            print(i)
            i+=1
            threads.append(t)
    except RuntimeError as e:
        print(f"Reached maximum threads: {len(threads)}")
    finally:
        for t in threads:
            t.join()

test_max_threads()
