fail() {
    echo "$1" >&2
    exit 1
}

MODEL_NAME=""
EXPERIMENT_NAME=""
PEFT=""
NUM_SHOTS=0
EVAL_DEV=false
QPEFT=false
MERGE_PEFT=true

while [[ "$1" = --* ]]; do
    option="${1#\--}"
    save=""
    case $option in
        model_name)
            save="MODEL_NAME"
            ;;
        experiment_name)
            save="EXPERIMENT_NAME"
            ;;
        peft)
            save="PEFT"
            ;;
        num_shots)
            save="NUM_SHOTS"
            ;;
        eval_dev)
            EVAL_DEV=true
            ;;
        qpeft)
            QPEFT=true
            ;;
        no_merge)
            MERGE_PEFT=false
            ;;
        *)
            fail "Invalid option: $1"
            ;;
    esac

    shift

    if [[ "$save" != "" ]]; then
        if [[ $# == 0 ]]; then
            fail "Missing value for option --$option"
        fi
        eval $save=\""$1"\"
        shift
    fi
done

if [[ $# != 0 ]]; then
    fail "Unxepected or extraneous parameter(s) detected: $@"
fi

if [[ "$EXPERIMENT_NAME" == "" ]]; then
    fail "Missing required argument --experiment_name"
fi

if [[ "$MODEL_NAME" == "" ]]; then
    fail "Missing required argument --model_name"
fi

PEFT_ARG=""
if [[ "$PEFT" != "" ]]; then
    PEFT_ARG="--peft_name_or_path $PEFT"
fi

EVAL_DEV_ARG=""
if [[ "$EVAL_DEV" == true ]]; then
    if [[ "$DEV_AVAILABLE" == false ]]; then
        fail "No dev set available for task $TASK_NAME"
    fi
    EVAL_DEV_ARG="--eval_dev"
fi

QPEFT_ARG=""
if [[ "$QPEFT" == true ]]; then
    QPEFT_ARG="--load_in_4bit --bnb_4bit_use_double_quant"
fi

MERGE_ARG=""
if [[ "$MERGE_PEFT" == false ]]; then
    MERGE_ARG="--no_merge_peft"
fi

message="Experiment $EXPERIMENT_NAME: evaluating $MODEL_NAME on $TASK_NAME"
if [[ "$EVAL_DEV" == true ]]; then
    message="$message dev"
else
    message="$message test"
fi
message="$message set in a ${NUM_SHOTS}-shot setting"
if [[ "$PEFT" != "" ]]; then
    message="$message with PEFT $PEFT, which will"
    if [[ "$MERGE_PEFT" == false ]]; then
        message="$message NOT"
    fi
    message="$message be merged into the base model"
fi

echo "$message" >&2

