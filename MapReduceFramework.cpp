#include <atomic>
#include <csignal>
#include <condition_variable>
#include <algorithm>
#include <unistd.h>
#include <malloc.h>
#include "MapReduceFramework.h"
#include "Barrier.h"


///------------------------------ STRUCTS -----------------------------------------
//struct JobHandleObject {
//    JobHandleObject(JobState *job_state, pthread_t *threads, int number_of_threads) {
//        this->job_state = job_state;
//        this->threads = threads;
//        this->number_of_threads = number_of_threads;
//        this->flag.store(0);
//    }
//
//    JobState *job_state;
//    std::atomic<int> flag;
//    pthread_t *threads;
//    std::mutex wait_mutex;
//    int number_of_threads;
//
//    void activate_flag() {
//        this->flag.store(1);
//    }
//
//    bool get_flag() {
//        return this->flag.load();
//    }
//};

struct JobContext { // resources used by all threads - every thread hold a pointer
    pthread_t *threads;

    JobState *job_state;
    int number_of_threads;
    std::vector<IntermediateVec *> *all_intermediate_vec;
    std::vector<IntermediateVec *> *shuffled_intermediate_vec;
    const MapReduceClient *client;
    OutputVec *output_vec;
    std::atomic<int> *map_atomic_counter;
    std::atomic<int> *reduce_atomic_counter;
    Barrier *barrier;
    std::atomic<int> flag;
    pthread_mutex_t wait_mutex;
    pthread_mutex_t end_map_stage_mutex;
    pthread_mutex_t update_stage_mutex;
    pthread_mutex_t emit3_mutex;
    pthread_mutex_t reduce_stage_mutex;
    float total_items_to_process;
    std::atomic<uint64_t> *current_processed_atomic_counter;

    JobContext(JobState * job_state, const MapReduceClient *client, OutputVec *output_vec, int numOfThreads) {
        this->client = client;
        this->all_intermediate_vec = new std::vector<IntermediateVec *>();
        this->shuffled_intermediate_vec = new std::vector<IntermediateVec *>();
        this->output_vec = output_vec;
        this->job_state = job_state;
        this->barrier = new Barrier(numOfThreads);
        this->total_items_to_process = 0;

        // init atomic counters
        this->map_atomic_counter = new std::atomic<int>();
        this->reduce_atomic_counter = new std::atomic<int>();
        this->current_processed_atomic_counter = new std::atomic<uint64_t>();
        this->map_atomic_counter->store(0);
        this->reduce_atomic_counter->store(0);
        this->current_processed_atomic_counter->store(0);

        //init mutexes
        this->wait_mutex = PTHREAD_MUTEX_INITIALIZER;
        this->end_map_stage_mutex = PTHREAD_MUTEX_INITIALIZER;
        this->update_stage_mutex = PTHREAD_MUTEX_INITIALIZER;
        this->reduce_stage_mutex = PTHREAD_MUTEX_INITIALIZER;
        this->emit3_mutex = PTHREAD_MUTEX_INITIALIZER;
    }

    ~JobContext(){
        delete this->all_intermediate_vec;
        delete this->shuffled_intermediate_vec;
        delete this->barrier;
        delete this->map_atomic_counter;
        delete this->reduce_atomic_counter;
        delete this->current_processed_atomic_counter;
        pthread_mutex_destroy(&wait_mutex);
        pthread_mutex_destroy(&end_map_stage_mutex);
        pthread_mutex_destroy(&update_stage_mutex);
        pthread_mutex_destroy(&reduce_stage_mutex);
        pthread_mutex_destroy(&emit3_mutex);

    }

    void activate_flag() {
        this->flag.store(1);
    }

    bool get_flag() {
        return this->flag.load();
    }

};

struct ThreadContext {
    int thread_id;
    const InputVec *input_vec;
    JobContext *job_context; // JobContext of all threads
    IntermediateVec *intermediate_vec;

    ThreadContext(){
        this->thread_id = -1;
        this->intermediate_vec = new IntermediateVec();
        this->job_context = nullptr;
        this->input_vec = nullptr;
    }

    ThreadContext(int tid, const InputVec &inputVec, JobContext* job_context){
        this->thread_id = tid;
        this->job_context = job_context;
        this->input_vec = &inputVec;
        this->intermediate_vec = new IntermediateVec();
    }
    ~ThreadContext(){
        delete this->intermediate_vec;
    }
};

///------------------------------ FUNCTIONS USED -----------------------------------------

// init funcs
JobState *get_new_job_state();

ThreadContext **init_thread_contexts(OutputVec &outputVec, int multiThreadLevel, const MapReduceClient &client,
                                    const InputVec &inputVec, JobContext *job_context);

void init_thread_context(const InputVec &inputVec, ThreadContext *thread_context, JobContext *job_context, int i);


// helpers for shuffle funcs
K2 *get_max_key(std::vector<IntermediateVec *> *all_vectors);

void
pop_all_max_keys(K2 *max_key, IntermediateVec *intermediateVecOutput, std::vector<IntermediateVec *> *all_vec_input,
                 ThreadContext *tc, int &current_number_of_processed_pairs);

void remove_empty_vectors(std::vector<IntermediateVec *> *all_vectors, ThreadContext *pContext);


void release_all_resources(JobHandle job_handle);

void update_stage(JobContext *job_context, stage_t stage, float i);

int get_size_of_vector_of_vectors(std::vector<IntermediateVec *> *pVector);

/**
 *
 * @param a IntermediatePair
 * @param b IntermediatePair
 * @return true if (key in a) < (key in b)
 */
bool compare(const IntermediatePair &a, const IntermediatePair &b) {
    return *(a.first) < *(b.first);
}

// map reduce
void *map_reduce_method(void *context) {
    auto *tc = (ThreadContext *) context;

    //Map
    auto input_vec_size = (float) tc->input_vec->size();
    update_stage(tc->job_context, MAP_STAGE, input_vec_size);
    int current_index = (*(tc->job_context->map_atomic_counter))++;
    tc->job_context->total_items_to_process = input_vec_size;
    while (current_index < input_vec_size) {
        //Todo: how should I increase the percentage
        tc->job_context->client->map(tc->input_vec->at(current_index).first, tc->input_vec->at(current_index).second,
                                     context);
        (*(tc->job_context->current_processed_atomic_counter))++;
        current_index = (*(tc->job_context->map_atomic_counter))++;
    }
    //Sort
    printf("Before barriers: %d\n", tc->thread_id);
    std::sort(tc->intermediate_vec->begin(), tc->intermediate_vec->end(), compare);
    tc->job_context->barrier->barrier();
    printf("Between barriers: %d\n", tc->thread_id);

    //Shuffle
    if (tc->thread_id == 0) {
        int all_intermediate_vec_size = get_size_of_vector_of_vectors(tc->job_context->all_intermediate_vec);
        update_stage(tc->job_context, SHUFFLE_STAGE, all_intermediate_vec_size);
        std::vector<IntermediateVec *> *all_intermediate_vec = tc->job_context->all_intermediate_vec;
        std::vector<IntermediateVec *> *shuffled_vector = tc->job_context->shuffled_intermediate_vec;
//        int current_number_of_processed_pairs = 0;
        while (!all_intermediate_vec->empty() && !all_intermediate_vec->at(0)->empty()) {
            K2 *max_key = get_max_key(all_intermediate_vec); // gets the maximal key of all keys imn all vec
            auto *max_key_vector = (IntermediateVec *) malloc(sizeof(IntermediateVec));
            pop_all_max_keys(max_key, max_key_vector, all_intermediate_vec, tc, all_intermediate_vec_size);
            shuffled_vector->push_back(max_key_vector);
        }

    }


    tc->job_context->barrier->barrier();
    //Reduce
    pthread_mutex_lock(&tc->job_context->end_map_stage_mutex);
    if(tc->thread_id ==0){
        float number_of_shuffled_items = get_size_of_vector_of_vectors(tc->job_context->shuffled_intermediate_vec);
        update_stage(tc->job_context, REDUCE_STAGE, number_of_shuffled_items);
    }
    pthread_mutex_unlock(&tc->job_context->end_map_stage_mutex);
    int current_reduce_index = (*(tc->job_context->reduce_atomic_counter))++;
    unsigned long shuffled_intermediate_vec_size = tc->job_context->shuffled_intermediate_vec->size();
    while (current_reduce_index < shuffled_intermediate_vec_size) {
//        pthread_mutex_lock(&tc->job_context->reduce_stage_mutex);
        tc->job_context->client->reduce(tc->job_context->shuffled_intermediate_vec->at(current_reduce_index), context);
        (*(tc->job_context->current_processed_atomic_counter)).fetch_add(tc->job_context->shuffled_intermediate_vec->at(current_reduce_index)->size());
//        pthread_mutex_unlock(&tc->job_context->reduce_stage_mutex);
        current_reduce_index = (*(tc->job_context->reduce_atomic_counter))++;
    }
    printf("After barriers: %d\n", tc->thread_id);
    return tc->job_context;
}

int get_size_of_vector_of_vectors(std::vector<IntermediateVec *> *pVector) {
    int sum = 0;
    for (const auto &innerVec: *pVector) {
        sum += innerVec->size();
    }
    return sum;
}

//void update_stage(JobState *job_stage, stage_t stage) {
//
//    job_stage->stage = stage;
//    job_stage->percentage = 0;
//}
void update_stage(JobContext *jc, stage_t stage, float total) {

    //TODO: ADD MUTEX as these are many atomic functions together
    jc->job_state->stage = stage;
    jc->job_state->percentage = 0;
    jc->total_items_to_process = total;
    jc->current_processed_atomic_counter->store(0);
}

/**
 * goes over all vectors and checks only at the back cell for max value (as the cells are sorted)
 * @param all_vectors after sort phase
 * @return value of the maximal key found
 */
K2 *get_max_key(std::vector<IntermediateVec *> *all_vectors) {
    K2 *max_key = (K2 *) malloc(sizeof(K2));
    bool is_first_iteration = true;
    for (auto vec: *all_vectors) {
        if (is_first_iteration) {
            max_key = vec->back().first;
            is_first_iteration = false;
            continue;
        }
//        max_key = std::max(max_key, vec->back().first);
        if (!vec->empty() && *max_key < *(vec->back().first)) {
            max_key = vec->back().first;
        }
    }
    return max_key;
}

/**
 *
 * @param max_key found after going over all vec.back()
 * @param intermediateVecOutput the output vector to place the result
 * @param all_vec_input all vectors
 */
void
pop_all_max_keys(K2 *max_key, IntermediateVec *intermediateVecOutput, std::vector<IntermediateVec *> *all_vec_input,
                 ThreadContext *tc, int &current_number_of_processed_pairs) {
    for (auto vec: *all_vec_input) {
        while (!vec->empty() && !(*vec->back().first < *max_key) &&
               !(*max_key < *vec->back().first)) { // goes over the vector backwards as long as it equals the max_key
            intermediateVecOutput->push_back(vec->back());
            vec->pop_back();
            (*(tc->job_context->current_processed_atomic_counter))++;
//            tc->job_context->job_state->percentage =
//                    ((float) current_number_of_processed_pairs / (float) (initial_size)) * 100;
        }
    }
    //after finishing emptying every vector from max_key pairs - going over and deleting all empty vectors
    remove_empty_vectors(all_vec_input, tc);
}

/**
 * frees the vec alloc when empty, and returns true. otherwise false without free
 * @param intermediate_vec out of all vectors - to be freed when empty
 * @return true if it was empty, otherwise false (without deleting)
 */
bool free_if_empty(const IntermediateVec *intermediate_vec) {
    bool flag = intermediate_vec->empty();
    if (flag) {
        delete intermediate_vec; //only if empty!
    }
    return flag;
}

/**
 * Remove empty vectors of all vectors
 * @param all_vectors after sort phase
 */
void remove_empty_vectors(std::vector<IntermediateVec *> *all_vectors, ThreadContext *pContext) {
    if (all_vectors->size() != 1) {
        bool (*check_if_vector_empty)(const IntermediateVec *) = [](const IntermediateVec *intermediate_vec) {
            return free_if_empty(
                    intermediate_vec);
        };
        const std::vector<IntermediateVec *>::iterator &vectors_to_erase = std::remove_if(all_vectors->begin(),
                                                                                          all_vectors->end(),
                                                                                          check_if_vector_empty);
        all_vectors->erase(
                vectors_to_erase, all_vectors->end());
    }
}

/**
 *
 * @param outputVec
 * @param multiThreadLevel num of threads to create
 * @param client
 * @param inputVec
 * @param job_state
 * @param job_context shared resources
 * @return
 */
ThreadContext** init_thread_contexts(OutputVec &outputVec, int multiThreadLevel, const MapReduceClient &client,
                                    const InputVec& inputVec, JobContext *job_context) {
    auto **thread_contexts = (ThreadContext **) malloc(multiThreadLevel * sizeof(ThreadContext*));
//    ThreadContext * thread_contexts[multiThreadLevel];

    for (int i = 0; i < multiThreadLevel; ++i) {
//        init_thread_context(inputVec, thread_contexts + i, job_context, i);
        thread_contexts[i] = new ThreadContext(i,inputVec,job_context);
        job_context->all_intermediate_vec->push_back(thread_contexts[i]->intermediate_vec);
    }
    return thread_contexts;
}
/**
 *
 * @param inputVec of all pairs before map phase
 * @param thread_context
 * @param job_context pointer to shared resources
 * @param i index in threads
 */
void init_thread_context(const InputVec &inputVec, ThreadContext *thread_context,
                         JobContext *job_context, const int i) {
//    thread_context = new ThreadContext();
//    thread_context->thread_id = i;
//    thread_context->job_context = job_context;
//    thread_context->input_vec = &inputVec;
//    thread_context->intermediate_vec = new IntermediateVec();
    thread_context = new ThreadContext(i,inputVec,job_context);
    job_context->all_intermediate_vec->push_back(thread_context->intermediate_vec);
}

/**
 *
 * @return JobState obj where stage = UNDEFINED_STAGE
 */
JobState *get_new_job_state() {
    auto *job = (JobState *) malloc(sizeof(JobState));
    job->stage = UNDEFINED_STAGE;
    job->percentage = 0;
    return job;
}


///------------------------------ LIBRARY -----------------------------------------

/**
 * starts running the MapReduce algorithm (w several threads) and returns a job handler
 * You can assume that the input to this function is valid
 * @param client the task that the framework should run (The implementation of MapReduceClient)
 * @param inputVec a vector of type std::vector<std::pair<K1*, V1*>>, the input elements
 * @param output_vec a vector of type std::vector<std::pair<K3*, V3*>>,
 *                  to which the output elements will be added before returning.
 *                  You can assume that output_vec is empty.
 * @param multiThreadLevel the number of worker threads to be used for running the algorithm
 *                  You can assume multiThreadLevel argument is valid (greater or equal to 1)
 * @return JobHandle that will be used for monitoring the job
 */
JobHandle
startMapReduceJob(const MapReduceClient &client, const InputVec &inputVec, OutputVec &output_vec, int multiThreadLevel) {
    pthread_t threads[multiThreadLevel];
    JobState *job_state = get_new_job_state();
    auto* job_context = new JobContext(job_state,&client,&output_vec,multiThreadLevel);
    ThreadContext **thread_contexts = init_thread_contexts(output_vec, multiThreadLevel, client, inputVec, job_context);
    job_context->threads = threads;
    for (int i = 0; i < multiThreadLevel; ++i) {
        pthread_create(threads + i, nullptr, map_reduce_method, *(thread_contexts + i));
    }
//
//    int counter = 0;
//    for (const auto &pair: inputVec) {
//        client.map(pair.first, pair.second, thread_contexts + counter);
//        counter++;
//    }

    return static_cast<JobHandle>(job_context);
}

/**
 * gets a JobHandle and updates the state of the job into the given JobState struct
 * @param job JobHandle
 * @param state JobState struct to update the job to
 */
void getJobState(JobHandle job, JobState *state) {
    auto *job_context = static_cast<JobContext *>(job);
    pthread_mutex_lock(&job_context->update_stage_mutex); //TODO make sure mutex work
    state->stage = job_context->job_state->stage;
    float current_processed = (float) job_context->current_processed_atomic_counter->load();
    state->percentage = (current_processed / job_context->total_items_to_process) * 100;
    pthread_mutex_unlock(&job_context->update_stage_mutex);
}

/**
 * Releasing all resources of a job.
 * After this function is called the job handle will be invalid.
 * In case that the function is called and the job is not finished yet wait until the job is finished to close it
 * @param job JobHandle
 */
void closeJobHandle(JobHandle job) {
    waitForJob(job);
    auto *job_context = static_cast<JobContext *>(job);
    delete job_context;
}

/**
 * produces a (K2*, V2*) pair.
 * updates the number of intermediary elements using atomic counter
 * @param key of input intermediary element
 * @param value of input intermediary element
 * @param context data structure of the thread that created the intermediary element
 */
void emit2(K2 *key, V2 *value, void *context) {
    auto *tc = (ThreadContext *) context;
    IntermediatePair pair = IntermediatePair(key, value);
    tc->intermediate_vec->push_back(pair);
}

/**
 * produces a (K3*, V3*) pair.
 * saves the output element in the context data structures
 * updates the number of output elements using atomic counter (done before after every reduce)
 * @param key of input output element
 * @param value of input output element
 * @param context data structure of the thread that created the output element
 */
void emit3(K3 *key, V3 *value, void *context) {
    auto *tc = (ThreadContext *) context;
    pthread_mutex_lock(&tc->job_context->emit3_mutex);
    OutputPair pair = OutputPair(key, value);
    tc->job_context->output_vec->push_back(pair);
    pthread_mutex_unlock(&tc->job_context->emit3_mutex);
}

/**
 * gets the JobHandle returned by startMapReduceFramework and waits until it is finished
 * @param job the JobHandle returned by startMapReduceFramework
 */
void waitForJob(JobHandle job) {
    auto *job_context = static_cast<JobContext *>(job);
    pthread_mutex_lock(&job_context->wait_mutex);
    if (!job_context->get_flag()) {
        job_context->activate_flag();
        for (int i = 0; i < job_context->number_of_threads; ++i) {
            pthread_join(job_context->threads[i], nullptr);
        }
    }
    pthread_mutex_unlock(&job_context->wait_mutex);
};