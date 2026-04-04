<template>
  <div id="layout" :class="{ 'no-earth-bg': isHomePage }">
    <NavBar />
    <main class="main-content">
      <RouterView v-slot="{ Component, route }">
        <Transition name="page">
          <component :is="Component" :key="route.path" />
        </Transition>
      </RouterView>
    </main>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import { useRoute } from 'vue-router'
import NavBar from '@/components/NavBar.vue'

const route = useRoute()
const isHomePage = computed(() => route.path === '/')
</script>

<style>
#layout {
  display: flex;
  flex-direction: column;
  min-height: 100vh;
  background:
    linear-gradient(180deg, rgba(3,13,26,0.15) 0%, rgba(3,13,26,0.55) 100%),
    url('/earth-night.jpg') center center / cover no-repeat fixed;
}
#layout.no-earth-bg {
  background: var(--bg);
}

.main-content {
  flex: 1;
  display: flex;
  flex-direction: column;
  padding-top: var(--nav-h);
  position: relative;
}

/* Page transition — synchronized fade for bg + content */
.page-enter-active {
  transition: opacity 0.4s ease;
}
.page-leave-active {
  transition: opacity 0.4s ease;
}
.page-enter-from {
  opacity: 0;
}
.page-leave-to {
  opacity: 0;
}
</style>
