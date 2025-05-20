import json
import time
import boto3
from typing import Dict, Any, List, Optional
from mcp_client import MCPClient


class BedrockMCPClient:
    """Bedrock과 통합된 MCP 클라이언트"""

    def __init__(self, mcp_url: str, region: str = None, auth_token: str = None, model_id: str = None,
                 session_id: str = None):
        """
        Bedrock MCP 클라이언트 초기화

        Args:
            mcp_url: MCP 서버 URL
            region: AWS 리전 (기본값: Lambda 함수의 리전)
            auth_token: MCP 인증 토큰 (선택 사항)
            model_id: Bedrock 모델 ID (기본값: 'anthropic.claude-3-haiku-20240307-v1:0')
            session_id: 기존 세션 ID (선택 사항)
        """
        self.mcp_client = MCPClient(mcp_url, auth_token, session_id)
        self.region = region or boto3.session.Session().region_name
        self.model_id = model_id or 'anthropic.claude-3-haiku-20240307-v1:0'
        self.bedrock_client = boto3.client('bedrock-runtime', region_name=self.region)
        self.tools = []
        self.messages = []

    def initialize(self) -> str:
        """
        MCP 세션 초기화 및 도구 목록 로드

        Returns:
            세션 ID
        """
        session_id = self.mcp_client.initialize()
        self.tools = self.mcp_client.list_tools()
        return session_id

    def invoke_with_tools(self, prompt: str, system_prompt: str = None) -> Dict[str, Any]:
        """
        MCP 도구를 사용하여 Bedrock 모델 호출

        Args:
            prompt: 사용자 프롬프트
            system_prompt: 시스템 프롬프트 (선택 사항)

        Returns:
            Bedrock 모델 응답
        """
        # 세션 및 도구가 초기화되지 않은 경우
        if not self.tools:
            self.initialize()

        # 메시지 추가
        self.messages.append({
            'role': 'user',
            'content': [{'text': prompt}]
        })

        # Bedrock 도구 형식으로 변환
        bedrock_tools = []
        for tool in self.tools:
            bedrock_tools.append({
                'toolSpec': {
                    'name': tool['name'].replace('-', '_'),  # 대시를 언더스코어로 변환 (Bedrock 요구사항)
                    'description': tool['description'],
                    'inputSchema': {
                        'json': {
                            'type': tool['inputSchema'].get('type', 'object'),
                            'properties': tool['inputSchema'].get('properties', {}),
                            'required': tool['inputSchema'].get('required', [])
                        }
                    }
                }
            })

        # Bedrock 요청 구성
        request = {
            'modelId': self.model_id,
            'messages': self.messages,
            'toolConfig': {
                'tools': bedrock_tools,
                'toolChoice': {'auto': {}}
            }
        }

        # 시스템 프롬프트가 제공된 경우 추가
        if system_prompt:
            request['system'] = [{'text': system_prompt}]

        # 모델 호출 루프 시작 (도구 사용이 완료될 때까지)
        final_response = None
        max_iterations = 5  # 무한 루프 방지를 위한 최대 반복 횟수
        iteration = 0

        while iteration < max_iterations:
            iteration += 1

            # Bedrock 모델 호출
            response = self.bedrock_client.converse(**request)

            # 응답 파싱
            stop_reason = response.get('stopReason')
            output_message = response.get('output', {}).get('message', {})

            # 메시지에 응답 추가
            if output_message:
                self.messages.append(output_message)

            # 도구 사용 요청인 경우
            if stop_reason == 'tool_use':
                # 도구 사용 블록 추출
                tool_uses = []
                for item in output_message.get('content', []):
                    if 'toolUse' in item:
                        tool_uses.append(item['toolUse'])

                # 각 도구에 대해 MCP 도구 호출
                tool_results = []
                for tool_use in tool_uses:
                    try:
                        # 도구 이름을 MCP 형식으로 변환 (언더스코어를 대시로)
                        mcp_tool_name = tool_use['name'].replace('_', '-')

                        # MCP 도구 호출
                        result = self.mcp_client.call_tool(mcp_tool_name, tool_use['input'])

                        # 도구 결과 형식화
                        tool_results.append({
                            'toolResult': {
                                'toolUseId': tool_use['toolUseId'],
                                'content': [{'text': json.dumps(result)}],
                                'status': 'success'
                            }
                        })
                    except Exception as e:
                        # 오류 처리
                        tool_results.append({
                            'toolResult': {
                                'toolUseId': tool_use['toolUseId'],
                                'content': [{'text': f'도구 실행 오류: {str(e)}'}],
                                'status': 'error'
                            }
                        })

                # 도구 결과를 추가하여 계속 대화
                if tool_results:
                    self.messages.append({
                        'role': 'user',
                        'content': tool_results
                    })
                    continue  # 다음 반복으로

            # 대화 종료 또는 다른 종료 이유
            elif stop_reason in ['end_turn', 'stop_sequence']:
                final_response = response
                break

            # 토큰 제한에 도달한 경우 계속하도록 요청
            elif stop_reason == 'max_tokens':
                self.messages.append({
                    'role': 'user',
                    'content': [{'text': 'Please continue.'}]
                })
                continue

            # 알 수 없는 종료 이유
            else:
                raise Exception(f'알 수 없는 종료 이유: {stop_reason}')

        return final_response

    def process_user_input(self, user_input: str, system_prompt: str = None) -> str:
        """
        사용자 입력 처리 및 최종 텍스트 응답 반환

        Args:
            user_input: 사용자 질문/입력
            system_prompt: 시스템 프롬프트 (선택 사항)

        Returns:
            최종 텍스트 응답
        """
        response = self.invoke_with_tools(user_input, system_prompt)

        # 응답에서 텍스트 추출
        output_message = response.get('output', {}).get('message', {})
        content = output_message.get('content', [])

        # 첫 번째 텍스트 콘텐츠 항목 찾기
        for item in content:
            if 'text' in item:
                return item['text']

        return "응답을 생성할 수 없습니다."

    def close(self) -> bool:
        """
        MCP 세션 종료

        Returns:
            성공 여부
        """
        return self.mcp_client.close()