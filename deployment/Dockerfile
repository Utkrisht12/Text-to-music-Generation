FROM amazon/aws-lambda-python:3.9

ARG ACTUAL_LAMBDA
ARG LAMBDA_ENTRYPOINT

COPY ./lambda-entrypoint.sh /lambda-entrypoint.sh
COPY ./${ACTUAL_LAMBDA}/requirements.txt ${LAMBDA_TASK_ROOT}/${ACTUAL_LAMBDA}/requirements.txt

RUN pip install --upgrade pip
RUN pip3 install -r ${LAMBDA_TASK_ROOT}/${ACTUAL_LAMBDA}/requirements.txt --target "${LAMBDA_TASK_ROOT}"

COPY ./utils/. ${LAMBDA_TASK_ROOT}/utils
COPY ./${ACTUAL_LAMBDA}/. ${LAMBDA_TASK_ROOT}/${ACTUAL_LAMBDA}

ENV LAMBDA_FULL_ENTRYPOINT "${ACTUAL_LAMBDA}.app.${LAMBDA_ENTRYPOINT}"
CMD ["$LAMBDA_FULL_ENTRYPOINT"]
