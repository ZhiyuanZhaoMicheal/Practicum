import { createRouter, createWebHashHistory } from 'vue-router'

const routes = [
  {
    path: '/',
    name: 'Home',
    component: () => import('@/views/HomeView.vue'),
    meta: { title: 'NightLight — Overview' },
  },
  {
    path: '/map',
    name: 'Map',
    component: () => import('@/views/MapView.vue'),
    meta: { title: 'NightLight — Interactive Map' },
  },
  {
    path: '/charts',
    name: 'Charts',
    component: () => import('@/views/ChartsView.vue'),
    meta: { title: 'NightLight — Results' },
  },
  {
    path: '/docs',
    name: 'Docs',
    component: () => import('@/views/DocsView.vue'),
    meta: { title: 'NightLight — Technical Documentation' },
  },
  {
    path: '/docs/:section',
    name: 'DocsDetail',
    component: () => import('@/views/DocsDetailView.vue'),
    meta: { title: 'NightLight — Documentation' },
  },
]

const router = createRouter({
  history: createWebHashHistory(),
  routes,
  scrollBehavior(to) {
    if (to.hash) {
      return { el: to.hash, behavior: 'smooth', top: 72 }
    }
    return { top: 0 }
  },
})

// Update page title on route change
router.afterEach(route => {
  document.title = route.meta?.title || 'NightLight'
})

export default router
