const PHYSICAL_AI_STUDIO = 'physical-ai-studio';

const extensions = [
  {
    type: 'app.navigation/section',
    properties: {
      id: PHYSICAL_AI_STUDIO,
      title: 'Physical AI studio',
      group: '4_physical_ai_studio',
      iconRef: () => import('./PhysicalAiStudioNavIcon'),
    },
  },
  {
    type: 'app.navigation/href',
    properties: {
      id: 'robotics-playground',
      title: 'Robotics Playground',
      href: '/physicalAiStudio/roboticsPlayground',
      section: PHYSICAL_AI_STUDIO,
      path: '/physicalAiStudio/roboticsPlayground/*',
      label: 'Experimental',
    },
  },
  {
    type: 'app.route',
    properties: {
      path: '/physicalAiStudio/roboticsPlayground/*',
      component: () => import('./RoboticsPlayground'),
    },
  },
];

export default extensions;
