from __future__ import annotations

from fastapi import APIRouter
from fastapi import status
from fastapi.encoders import jsonable_encoder

from api.helpers.exception_handler import ExceptionHandler
from api.helpers.exception_handler import ResponseMessage
from api.models.text2image import T2IInput
from api.models.text2image import T2IOutput
from infra.load_model import Text2ImageInput
from infra.load_model import Text2ImageService
from shared.logging import get_logger
from shared.settings import Settings

Text2Image = APIRouter(prefix='/v1')
logger = get_logger(__name__)
settings = Settings()

try:
    text2image_model = Text2ImageService(settings=settings)
    logger.info('Init T2I service success!')
except Exception as e:
    logger.error(f'Error initializing text to image service: {str(e)}')
    raise e


@Text2Image.post(
    '/Text2Image',
    response_model=T2IOutput,
    responses={
        status.HTTP_200_OK: {
            'content': {
                'application/json': {
                    'example': {
                        'message': ResponseMessage.SUCCESS,
                        'info': {
                            'status': True,
                            'image_base64': '<base64_encoded_image>',
                        },
                    },
                },
            },
        },
        status.HTTP_400_BAD_REQUEST: {
            'description': 'Bad Request',
            'content': {
                'application/json': {
                    'example': {
                        'message': ResponseMessage.BAD_REQUEST,
                    },
                },
            },
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            'description': 'Internal Server Error',
            'content': {
                'application/json': {
                    'example': {
                        'message': ResponseMessage.INTERNAL_SERVER_ERROR,
                    },
                },
            },
        },
    },
)
async def t2i_service(inputs: T2IInput) -> T2IOutput:

    exception = ExceptionHandler(
        logger=logger.bind(), service_name=__name__,
    )
    if inputs is None or not inputs.prompt:
        return exception.handle_bad_request(
            'Prompt is missing or empty.',
            jsonable_encoder(inputs),
        )

    try:
        result = text2image_model.process(
            Text2ImageInput(
                prompt=inputs.prompt,
            ),
        )
        logger.info('T2I processed prompt successfully')
        return exception.handle_success(
            jsonable_encoder({'image_base64': result.image}),
        )
    except Exception as e:
        logger.exception(
            f'Error processing T2I request: {str(e)}',
        )
        return exception.handle_exception(
            'Error during to text to image generation.',
            jsonable_encoder({}),
        )
