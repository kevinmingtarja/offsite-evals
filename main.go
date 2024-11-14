package main

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/hypermodeinc/modus/sdk/go/pkg/models"
	"github.com/hypermodeinc/modus/sdk/go/pkg/models/openai"
)

const modelName = "evaluator"

type Evaluation struct {
	Score     int    `json:"score"`
	Reasoning string `json:"reasoning"`
}

// taken from:
// https://github.com/flowaicom/flow-judge/blob/e56d199db4d79f184dac1e9ab2da83992acda14d/flow_judge/utils/prompt_formatter.py#L5
const promptFmt = `
# GOAL
Your job is to evaluate a task carried out by an AI system powered by a large
language model.

You will be provided with the inputs and output of the task, as well as the evaluation criteria
and scoring rubric. Your task is to evaluate the output of the AI system based on the evaluation
criteria and scoring rubric provided.

# INPUT
Below are the inputs required for performing the task:
<inputs>
%s
</inputs>

# OUTPUT
Below is the output of the task:
<output>
%s
</output>

# EVALUATION CRITERIA AND SCORING RUBRIC
Here are the evaluation criteria and the rubric that you need to use for evaluating the task:
<evaluation_criteria>
Based on the given context, evaluate how consistent and faithful the generated response is to the
context. The response should not contain any hallucinated or fabricated information that is not
supported by the context.
</evaluation_criteria>

<scoring_rubric>
- Score: 1: The response is completely inconsistent with the provided context. It contains significant amount
of hallucinated or fabricated information that directly contradicts or is not supported at all by
the context
- Score: 2: The response is mostly inconsistent with the provided context. While it may contain some
information from the context, it introduces a substantial amount of hallucinated or fabricated
details that deviate from the context
- Score: 3: The response is somewhat consistent with the provided context. It includes a mix of information
from the context and some hallucinated or fabricated details. The fabrications are minor and do
not significantly contradict the context
- Score: 4: The response is mostly consistent with the provided context. The vast majority of the content is
supported by the context, with only minor and inconsequential inconsistencies or fabrications, if
any
- Score: 5: The response is completely consistent with and faithful to the provided context. All details in
the response are directly supported by the context, without any hallucinated or fabricated
information
</scoring_rubric>

# INSTRUCTIONS FOR THE EVALUATION
1. Understand the task and criteria: Familiarize yourself with the task to be evaluated.
Review the evaluation criteria and scoring rubric to understand the different levels of
performance and the descriptions for each score.
2. Review the inputs and output: Look at the inputs provided for the task. Examine the output
generated from completing the task.
3. Compare output to score descriptions: Compare the output against the criteria and score
descriptions in the scoring rubric. For each criterion,decide which description best matches the
output.
4. After comparing the output to the score descriptions, pay attention to the small details that
might impact the final score that you assign. Sometimes a small difference can dictate the final
score.
5. Write verbal feedback justifying your evaluation that includes a detailed rationale, referring
to specific aspects of the output and comparing them to the rubric.
6. Assign a final score based on the scoring rubric.

## FORMAT FOR THE EVALUATION
- Write the verbal feedback inside <feedback> tags without any additional surrounding text.
- Write the numeric score inside <score> tags, without any additional surrounding text and always
after the feedback.

Please accurately evaluate the task. Strictly adhere to the evaluation criteria and rubric.
`

func ScoreResponse(input, output string) (*Evaluation, error) {
	model, err := models.GetModel[openai.ChatModel](modelName)
	if err != nil {
		return nil, err
	}

	in, err := model.CreateInput(
		openai.NewSystemMessage("You are a helpful assistant."),
		openai.NewUserMessage(fmt.Sprintf(promptFmt, input, output)),
	)
	if err != nil {
		return nil, err
	}
	in.Temperature = 0.5
	in.MaxTokens = 500

	out, err := model.Invoke(in)
	if err != nil {
		return nil, err
	}

	str := strings.TrimSpace(out.Choices[0].Message.Content)
	l1, r1 := strings.Index(str, "<feedback>\n"), strings.Index(str, "\n</feedback>")
	l2, r2 := strings.Index(str, "<score>\n"), strings.Index(str, "\n</score>")
	if l1 == -1 || r1 == -1 || l2 == -1 || r2 == -1 {
		return nil, fmt.Errorf("invalid response format")
	}
	reason := str[l1+len("<feedback>\n") : r1]
	score := str[l2+len("<score>\n") : r2]
	iscore, err := strconv.Atoi(score)
	if err != nil {
		return nil, err
	}

	return &Evaluation{
		Score:     iscore,
		Reasoning: reason,
	}, nil
}
